/*
 * ESP32-C6 Experimental CSI Receiver (Router Ping)
 * Upgraded to Thesis/Production Grade:
 * FreeRTOS Queues, Atomic Counters, AGC Lock, External Antenna, NVS Protection, TX Power Lock
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdatomic.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/event_groups.h"

#include "nvs_flash.h"
#include "esp_mac.h"
#include "rom/ets_sys.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_now.h"
#include "esp_task_wdt.h"
#include "driver/gpio.h"

#include "lwip/inet.h"
#include "lwip/netdb.h"
#include "lwip/sockets.h"
#include "ping/ping_sock.h"

#include "protocol_examples_common.h"
#include "esp_csi_gain_ctrl.h"

/* ── Ρυθμίσεις ──────────────────────────────────────────────────────────── */
#define CONFIG_SEND_FREQUENCY      100
#define TX_POWER_FIXED             72   // 18dBm (Κλειδωμένη ισχύς)
#define CONFIG_FORCE_GAIN          1    // AGC Lock για σταθερό ραντάρ
#define CSI_QUEUE_SIZE             30
#define CSI_BUF_MAX_LEN            512  // Ασφαλές μέγεθος για πακέτα Router

#define WIFI_ENABLE                GPIO_NUM_3
#define WIFI_ANT_CONFIG            GPIO_NUM_14

#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
#define ESP_IF_WIFI_STA ESP_MAC_WIFI_STA
#endif

static const char *TAG = "csi_router_exp";

/* Queue Struct */
typedef struct {
    uint32_t            rx_id;
    uint8_t             mac[6];
    wifi_pkt_rx_ctrl_t  rx_ctrl;
    float               compensate_gain;
    uint8_t             agc_gain;
    int8_t              fft_gain;
    int                 len;
    bool                first_word_invalid;
    int8_t              buf[CSI_BUF_MAX_LEN];
} csi_queue_item_t;

static QueueHandle_t s_csi_queue = NULL;
static atomic_uint s_drop_count = 0;
static uint32_t s_packet_counter = 0;

/* ── Αρχικοποιήσεις ─────────────────────────────────────────────────────── */
static void antenna_init(void) {
    gpio_set_direction(WIFI_ENABLE, GPIO_MODE_OUTPUT);
    gpio_set_level(WIFI_ENABLE, 0);
    vTaskDelay(pdMS_TO_TICKS(100));
    gpio_set_direction(WIFI_ANT_CONFIG, GPIO_MODE_OUTPUT);
    gpio_set_level(WIFI_ANT_CONFIG, 1);
    ESP_LOGI(TAG, "External antenna enabled");
}

/* ── Callback Λήψης (PRODUCER) ──────────────────────────────────────────── */
static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info) {
    if (!info || !info->buf) return;

    // Φιλτράρισμα: Δεχόμαστε πακέτα ΜΟΝΟ από το Router μας (BSSID)
    if (memcmp(info->mac, ctx, 6) != 0) return;

    /*  AGC GAIN LOCK (FORCE GAIN) */
    float compensate_gain = 1.0f;
    uint8_t agc_gain = 0;
    int8_t fft_gain = 0;
    static uint8_t s_agc_baseline = 0;
    static int8_t  s_fft_baseline = 0;
    static int s_gain_count = 0;

    esp_csi_gain_ctrl_get_rx_gain(&info->rx_ctrl, &agc_gain, &fft_gain);

    if (s_gain_count < 100) {
        esp_csi_gain_ctrl_record_rx_gain(agc_gain, fft_gain);
        s_gain_count++;
    } else if (s_gain_count == 100) {
        esp_csi_gain_ctrl_get_rx_gain_baseline(&s_agc_baseline, &s_fft_baseline);
        #if CONFIG_FORCE_GAIN
        esp_csi_gain_ctrl_set_rx_force_gain(s_agc_baseline, s_fft_baseline);
        #endif
        s_gain_count++;
    }
    esp_csi_gain_ctrl_get_gain_compensation(&compensate_gain, agc_gain, fft_gain);

    /* Δημιουργία Queue Item */
    csi_queue_item_t item = {
        .rx_id = s_packet_counter++, // Αύξων αριθμός (αφού δεν έχουμε Magic Header εδώ)
        .rx_ctrl = info->rx_ctrl,
        .compensate_gain = compensate_gain,
        .agc_gain = agc_gain,
        .fft_gain = fft_gain,
        .first_word_invalid = info->first_word_invalid,
    };
    memcpy(item.mac, info->mac, 6);
    item.len = (info->len <= CSI_BUF_MAX_LEN) ? info->len : CSI_BUF_MAX_LEN;
    memcpy(item.buf, info->buf, item.len);

    /*  NON-BLOCKING QUEUE PUSH ΜΕ ATOMIC OVERFLOW TRACKING */
    if (xQueueSend(s_csi_queue, &item, 0) != pdTRUE) {
        atomic_fetch_add(&s_drop_count, 1); 
    }
}

/* ── Εκτύπωση Δεδομένων (CONSUMER TASK) ─────────────────────────────────── */
static void csi_print_task(void *arg) {
    static bool s_header_printed = false;
    csi_queue_item_t item;

    while (true) {
        if (xQueueReceive(s_csi_queue, &item, portMAX_DELAY) != pdTRUE) continue;

        unsigned int drops = atomic_exchange(&s_drop_count, 0);
        if (drops > 0) {
            ESP_LOGW(TAG, "WARNING: %u CSI frames dropped! (UART Bottleneck)", drops);
        }

        if (!s_header_printed) {
            ESP_LOGI(TAG, "================ CSI RECV ROUTER ================");
            ets_printf("type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,channel,local_timestamp,sig_len,rx_state,len,first_word,data\n");
            s_header_printed = true;
        }

        // Ίδιο format με το ESP-to-ESP για να δουλεύει το python script
        ets_printf("CSI_DATA,%lu," MACSTR ",%d,%d,%d,%d,%d,%d,%lu,%lu,%d",
                   (unsigned long)item.rx_id, MAC2STR(item.mac),
                   item.rx_ctrl.rssi, item.rx_ctrl.rate, item.rx_ctrl.noise_floor,
                   item.fft_gain, item.agc_gain, item.rx_ctrl.channel,
                   item.rx_ctrl.timestamp, item.rx_ctrl.sig_len, item.rx_ctrl.rx_state);

        ets_printf(",%d,%d,\"[%d", item.len, item.first_word_invalid,
                   (int16_t)(item.compensate_gain * item.buf[0]));
        for (int i = 1; i < item.len; i++) {
            ets_printf(",%d", (int16_t)(item.compensate_gain * item.buf[i]));
        }
        ets_printf("]\"\n");
    }
}

/* ── Wi-Fi & PING Init ──────────────────────────────────────────────────── */
static void wifi_csi_init() {
    wifi_csi_config_t csi_config = {
        .enable                 = true,
        .acquire_csi_legacy     = true,
        .acquire_csi_ht20       = true,
        .acquire_csi_ht40       = true,
        .acquire_csi_su         = false,
        .acquire_csi_mu         = false,
        .acquire_csi_dcm        = false,
        .acquire_csi_beamformed = false,
        .acquire_csi_he_stbc    = 2,
        .val_scale_cfg          = false,
        .dump_ack_en            = false,
        .reserved               = false
    };
    
    static wifi_ap_record_t s_ap_info = {0};
    ESP_ERROR_CHECK(esp_wifi_sta_get_ap_info(&s_ap_info));
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    
    /* Περνάμε τη MAC (BSSID) του Router ως context για το φιλτράρισμα */
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, s_ap_info.bssid));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
}

static esp_err_t wifi_ping_router_start() {
    static esp_ping_handle_t ping_handle = NULL;

    esp_ping_config_t ping_config = ESP_PING_DEFAULT_CONFIG();
    ping_config.count             = 0;
    ping_config.interval_ms       = 1000 / CONFIG_SEND_FREQUENCY;
    ping_config.task_stack_size   = 3072;
    ping_config.data_size         = 1;

    esp_netif_ip_info_t local_ip;
    esp_netif_get_ip_info(esp_netif_get_handle_from_ifkey("WIFI_STA_DEF"), &local_ip);
    ESP_LOGI(TAG, "got ip:" IPSTR ", gw: " IPSTR, IP2STR(&local_ip.ip), IP2STR(&local_ip.gw));
    
    // Στοχεύουμε την IP της Πύλης (Gateway) που είναι το Router
    ping_config.target_addr.u_addr.ip4.addr = ip4_addr_get_u32(&local_ip.gw);
    ping_config.target_addr.type = ESP_IPADDR_TYPE_V4;

    esp_ping_callbacks_t cbs = { 0 };
    esp_ping_new_session(&ping_config, &cbs, &ping_handle);
    esp_ping_start(ping_handle);

    return ESP_OK;
}

/* ── Main ───────────────────────────────────────────────────────────────── */
void app_main(void) {
    /*  NVS WEAR-OUT PROTECTION */
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        esp_reset_reason_t reason = esp_reset_reason();
        if (reason == ESP_RST_PANIC || reason == ESP_RST_WDT) {
            ESP_LOGE(TAG, "Crash loop detected! Halting NVS erase.");
            while(1) { vTaskDelay(pdMS_TO_TICKS(1000)); }
        }
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    s_csi_queue = xQueueCreate(CSI_QUEUE_SIZE, sizeof(csi_queue_item_t));
    
    /*  ΑΥΞΗΜΕΝΟ STACK ΣΤΑ 8192 BYTES */
    xTaskCreate(csi_print_task, "csi_print", 8192, NULL, 5, NULL);

    antenna_init();

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    /*  ΠΡΟΣΟΧΗ: Απαιτεί ρύθμιση SSID/PASS στο idf.py menuconfig ⚠️ */
    ESP_ERROR_CHECK(example_connect());

    /*  TX POWER LOCK ΜΕΤΑ ΤΗ ΣΥΝΔΕΣΗ */
    ESP_ERROR_CHECK(esp_wifi_set_max_tx_power(TX_POWER_FIXED));

    wifi_csi_init();
    wifi_ping_router_start();

    esp_task_wdt_add(NULL);

    while (true) {
        esp_task_wdt_reset();
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}