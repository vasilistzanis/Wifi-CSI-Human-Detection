/*
 * ESP32-C6 Production Receiver (Thesis Grade)
 * Features: FreeRTOS Queue, AGC Lock (FORCE_GAIN), Magic Header Parser, 
 * Queue Overflow Metrics (Atomic/RISC-V), NVS Protection, HT40 Wi-Fi 4 Init Fix
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdatomic.h> // <--- Lock-Free Race Condition Handling

#include "esp_system.h"
#include "nvs_flash.h"
#include "esp_mac.h"
#include "rom/ets_sys.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_now.h"
#include "esp_csi_gain_ctrl.h"
#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_task_wdt.h"

#define WIFI_CHANNEL               11
#define TX_POWER_FIXED             72
#define CSI_QUEUE_SIZE             30
#define CONFIG_FORCE_GAIN          1    // ✅ LOCKED AGC ΓΙΑ ΣΤΑΘΕΡΟ ΡΑΝΤΑΡ

#define WIFI_ENABLE                GPIO_NUM_3
#define WIFI_ANT_CONFIG            GPIO_NUM_14

static const uint8_t KNOWN_SENDER_MAC[] = {0x1a, 0x00, 0x00, 0x00, 0x00, 0x00};
static const char *TAG = "csi_recv";

/* ✅ MAGIC HEADER CHECK */
#define RADAR_MAGIC_SIGNATURE 0xA1B2C3D4

/* Queue Struct */
#define CSI_BUF_MAX_LEN  256

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

/* ✅ ATOMIC DROPPED FRAMES COUNTER (RISC-V RV32IMAC 'A' Extension) */
static atomic_uint s_drop_count = 0;

/* ── Αρχικοποιήσεις ─────────────────────────────────────────────────────── */
static void antenna_init(void) {
    gpio_set_direction(WIFI_ENABLE, GPIO_MODE_OUTPUT);
    gpio_set_level(WIFI_ENABLE, 0);
    vTaskDelay(pdMS_TO_TICKS(100));
    gpio_set_direction(WIFI_ANT_CONFIG, GPIO_MODE_OUTPUT);
    gpio_set_level(WIFI_ANT_CONFIG, 1);
}

static void wifi_init(void) {
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    ESP_ERROR_CHECK(esp_netif_init());
    
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_start());

    /* ✅ FIX: Ενεργοποίηση πρωτοκόλλου 802.11n πριν ζητήσουμε το HT40 */
    esp_wifi_set_band_mode(WIFI_BAND_MODE_2G_ONLY);
    
    wifi_protocols_t protocols = {
        .ghz_2g = WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N
    };
    ESP_ERROR_CHECK(esp_wifi_set_protocols(WIFI_IF_STA, &protocols));

    // Τώρα το HT40 γίνεται δεκτό χωρίς Error 0x102
    wifi_bandwidths_t bandwidth = { .ghz_2g = WIFI_BW_HT40 };
    ESP_ERROR_CHECK(esp_wifi_set_bandwidths(WIFI_IF_STA, &bandwidth));
    
    ESP_ERROR_CHECK(esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_BELOW));
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    ESP_ERROR_CHECK(esp_wifi_set_mac(WIFI_IF_STA, KNOWN_SENDER_MAC));
    ESP_ERROR_CHECK(esp_wifi_set_max_tx_power(TX_POWER_FIXED));
}

static void wifi_esp_now_init(esp_now_peer_info_t peer) {
    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_set_pmk((uint8_t *)"pmk1234567890123"));
    esp_now_rate_config_t rate_config = {
        .phymode = WIFI_PHY_MODE_HT40,
        .rate    = WIFI_PHY_RATE_MCS0_LGI,
        .ersu = false, .dcm = false
    };
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));
    ESP_ERROR_CHECK(esp_now_set_peer_rate_config(peer.peer_addr, &rate_config));
}

/* ── Callback Λήψης (PRODUCER) ──────────────────────────────────────────── */
static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info) {
    if (!info || !info->buf) return;

    if (memcmp(info->mac, KNOWN_SENDER_MAC, 6) != 0) return;

    /* ✅ DYNAMIC PAYLOAD PARSING & FRAME FILTERING */
    uint32_t rx_id = 0;
    bool is_radar_packet = false;

    if (info->payload != NULL && info->payload_len >= 8) {
        for (int i = 0; i <= info->payload_len - 8; i++) {
            uint32_t check_magic;
            memcpy(&check_magic, info->payload + i, sizeof(uint32_t));
            
            if (check_magic == RADAR_MAGIC_SIGNATURE) {
                memcpy(&rx_id, info->payload + i + 4, sizeof(uint32_t));
                is_radar_packet = true;
                break;
            }
        }
    }

    if (!is_radar_packet) return; // Drop Management/Noise frames silently

    /* ✅ AGC GAIN LOCK (FORCE GAIN) */
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
        .rx_id = rx_id,
        .rx_ctrl = info->rx_ctrl,
        .compensate_gain = compensate_gain,
        .agc_gain = agc_gain,
        .fft_gain = fft_gain,
        .first_word_invalid = info->first_word_invalid,
    };
    memcpy(item.mac, info->mac, 6);
    item.len = (info->len <= CSI_BUF_MAX_LEN) ? info->len : CSI_BUF_MAX_LEN;
    memcpy(item.buf, info->buf, item.len);

    /* ✅ NON-BLOCKING QUEUE PUSH ΜΕ ATOMIC OVERFLOW TRACKING */
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

        /* ✅ Queue Overflow Warning Logging (Lock-Free) */
        unsigned int drops = atomic_exchange(&s_drop_count, 0);
        if (drops > 0) {
            ESP_LOGW(TAG, "WARNING: %u CSI frames dropped! (UART Bottleneck)", drops);
        }

        if (!s_header_printed) {
            ESP_LOGI(TAG, "================ CSI RECV ================");
            ets_printf("type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,channel,local_timestamp,sig_len,rx_state,len,first_word,data\n");
            s_header_printed = true;
        }

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

/* ── CSI Configuration ──────────────────────────────────────────────────── */
static void wifi_csi_init(void) {
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));
    wifi_csi_config_t csi_config = {
        .enable                 = true,
        .acquire_csi_legacy     = false,
        .acquire_csi_ht20       = true,
        .acquire_csi_ht40       = true,
        .acquire_csi_su         = true,
        .acquire_csi_mu         = true,
        .acquire_csi_dcm        = true,
        .acquire_csi_beamformed = true,
        .acquire_csi_he_stbc    = 2,
        .val_scale_cfg          = false,
        .dump_ack_en            = false,
        .reserved               = false
    };
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
}

/* ── Main ───────────────────────────────────────────────────────────────── */
void app_main(void) {
    /* ✅ NVS WEAR-OUT PROTECTION */
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        esp_reset_reason_t reason = esp_reset_reason();
        if (reason == ESP_RST_PANIC || reason == ESP_RST_WDT) {
            ESP_LOGE(TAG, "Crash loop detected! Halting NVS erase to protect flash memory.");
            while(1) { vTaskDelay(pdMS_TO_TICKS(1000)); }
        }
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    s_csi_queue = xQueueCreate(CSI_QUEUE_SIZE, sizeof(csi_queue_item_t));
    /* ✅ ΑΥΞΗΜΕΝΟ STACK ΣΤΑ 8192 BYTES */
    xTaskCreate(csi_print_task, "csi_print", 8192, NULL, 5, NULL);

    antenna_init();
    wifi_init();

    esp_now_peer_info_t peer = {
        .channel   = WIFI_CHANNEL,
        .ifidx     = WIFI_IF_STA,
        .encrypt   = false,
        .peer_addr = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff},
    };
    wifi_esp_now_init(peer);
    wifi_csi_init();

    esp_task_wdt_add(NULL);

    ESP_LOGI(TAG, "=========== CSI RECV READY ===========");

    while (true) {
        esp_task_wdt_reset();
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}