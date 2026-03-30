/*
 * ESP32-C6 Production Receiver (Thesis Grade) — Final
 * Features: FreeRTOS Queue, AGC Lock, Magic Header Parser,
 * Static Aligned Buffer Print, Atomic Drop Metrics,
 * NVS Protection, HT40 Fix, Queue/Task null checks,
 * Correct log silencing order, band_mode error check,
 * TX power verification, buffer overflow protection
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdatomic.h>

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

/* Fallback αν λείπουν MAC macros */
#ifndef MACSTR
#define MACSTR "%02x:%02x:%02x:%02x:%02x:%02x"
#define MAC2STR(a) (a)[0], (a)[1], (a)[2], (a)[3], (a)[4], (a)[5]
#endif

/* ── Ρυθμίσεις ──────────────────────────────────────────────────────────── */
#define WIFI_CHANNEL 11
#define TX_POWER_FIXED 72
#define CSI_QUEUE_SIZE 30
#define CONFIG_FORCE_GAIN 1
#define CSI_BUF_MAX_LEN 256
#define PRINT_BUF_SIZE 4096

#define WIFI_ENABLE GPIO_NUM_3
#define WIFI_ANT_CONFIG GPIO_NUM_14

#define RADAR_MAGIC_SIGNATURE 0xA1B2C3D4

static const uint8_t KNOWN_SENDER_MAC[] = {0x1a, 0x00, 0x00, 0x00, 0x00, 0x00};
static const char *TAG = "csi_recv";

/* ── Queue Struct ────────────────────────────────────────────────────────── */
typedef struct
{
    uint32_t rx_id;
    uint8_t mac[6];
    wifi_pkt_rx_ctrl_t rx_ctrl;
    float compensate_gain;
    uint8_t agc_gain;
    int8_t fft_gain;
    int len;
    bool first_word_invalid;
    int8_t buf[CSI_BUF_MAX_LEN];
} csi_queue_item_t;

static QueueHandle_t s_csi_queue = NULL;

/* ✅ Atomic drop counter — RISC-V RV32IMAC 'A' extension (lock-free) */
static atomic_uint s_drop_count = 0;

/* ════════════════════════════════════════════════════════════════════════
   1. ANTENNA INIT
   ════════════════════════════════════════════════════════════════════════ */
static void antenna_init(void)
{
    gpio_set_direction(WIFI_ENABLE, GPIO_MODE_OUTPUT);
    gpio_set_level(WIFI_ENABLE, 0);
    vTaskDelay(pdMS_TO_TICKS(100));
    gpio_set_direction(WIFI_ANT_CONFIG, GPIO_MODE_OUTPUT);
    gpio_set_level(WIFI_ANT_CONFIG, 1);
    ESP_LOGI(TAG, "External antenna enabled (GPIO3=LOW, GPIO14=HIGH)");
}

/* ════════════════════════════════════════════════════════════════════════
   2. WIFI INIT
   ════════════════════════════════════════════════════════════════════════ */
static void wifi_init(void)
{
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    ESP_ERROR_CHECK(esp_netif_init());

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_start());

    /* ✅ FIX: 802.11n πριν HT40 — αποτρέπει Error 0x102 */
    ESP_ERROR_CHECK(esp_wifi_set_band_mode(WIFI_BAND_MODE_2G_ONLY));

    wifi_protocols_t protocols = {
        .ghz_2g = WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N};
    ESP_ERROR_CHECK(esp_wifi_set_protocols(WIFI_IF_STA, &protocols));

    wifi_bandwidths_t bandwidth = {.ghz_2g = WIFI_BW_HT40};
    ESP_ERROR_CHECK(esp_wifi_set_bandwidths(WIFI_IF_STA, &bandwidth));

    ESP_ERROR_CHECK(esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_BELOW));
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));

    /* ✅ TX Power lock + verification */
    ESP_ERROR_CHECK(esp_wifi_set_max_tx_power(TX_POWER_FIXED));
    int8_t actual_power = 0;
    esp_wifi_get_max_tx_power(&actual_power);
    ESP_LOGI(TAG, "TX power locked at %.2f dBm", actual_power / 4.0f);
}

/* ════════════════════════════════════════════════════════════════════════
   3. ESP-NOW INIT
   ════════════════════════════════════════════════════════════════════════ */
static void wifi_esp_now_init(esp_now_peer_info_t peer)
{
    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_set_pmk((uint8_t *)"pmk1234567890123"));

    esp_now_rate_config_t rate_config = {
        .phymode = WIFI_PHY_MODE_HT40,
        .rate = WIFI_PHY_RATE_MCS0_LGI,
        .ersu = false,
        .dcm = false};
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));
    ESP_ERROR_CHECK(esp_now_set_peer_rate_config(peer.peer_addr, &rate_config));
}

/* ════════════════════════════════════════════════════════════════════════
   4. CSI CALLBACK — PRODUCER
   Τρέχει σε WiFi task context — πρέπει να είναι ΓΡΗΓΟΡΟ.
   Μόνο: check, copy, queue push. Ποτέ print.
   ════════════════════════════════════════════════════════════════════════ */
static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf)
        return;
    if (memcmp(info->mac, KNOWN_SENDER_MAC, 6) != 0)
        return;

    /* ✅ Magic Header search — safe memcpy για unaligned access */
    uint32_t rx_id = 0;
    bool is_radar_packet = false;

    if (info->payload != NULL && info->payload_len >= 8)
    {
        for (int i = 0; i <= (int)info->payload_len - 8; i++)
        {
            uint32_t check_magic;
            memcpy(&check_magic, info->payload + i, sizeof(uint32_t));
            if (check_magic == RADAR_MAGIC_SIGNATURE)
            {
                memcpy(&rx_id, info->payload + i + 4, sizeof(uint32_t));
                is_radar_packet = true;
                break;
            }
        }
    }
    if (!is_radar_packet)
        return;

    /* ✅ AGC Gain Lock */
    float compensate_gain = 1.0f;
    uint8_t agc_gain = 0;
    int8_t fft_gain = 0;

    static uint8_t s_agc_baseline = 0;
    static int8_t s_fft_baseline = 0;
    static int s_gain_count = 0;

    esp_csi_gain_ctrl_get_rx_gain(&info->rx_ctrl, &agc_gain, &fft_gain);

    if (s_gain_count < 100)
    {
        esp_csi_gain_ctrl_record_rx_gain(agc_gain, fft_gain);
        s_gain_count++;
    }
    else if (s_gain_count == 100)
    {
        esp_csi_gain_ctrl_get_rx_gain_baseline(&s_agc_baseline, &s_fft_baseline);
#if CONFIG_FORCE_GAIN
        esp_csi_gain_ctrl_set_rx_force_gain(s_agc_baseline, s_fft_baseline);
#endif
        s_gain_count++;
    }
    esp_csi_gain_ctrl_get_gain_compensation(&compensate_gain, agc_gain, fft_gain);

    /* Δημιουργία και αποστολή queue item */
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

    /* Non-blocking push — atomic count αν γεμάτο */
    if (xQueueSend(s_csi_queue, &item, 0) != pdTRUE)
    {
        atomic_fetch_add(&s_drop_count, 1);
    }
}

/* ════════════════════════════════════════════════════════════════════════
   5. CSI PRINT TASK — CONSUMER
   Εδώ επιτρέπεται ets_printf — δεν μπλοκάρει WiFi context.
   Static aligned buffer: χωρίς heap fragmentation, χωρίς malloc failure.
   ════════════════════════════════════════════════════════════════════════ */
static void csi_print_task(void *arg)
{
    static bool s_header_printed = false;
    csi_queue_item_t item;

    /* ✅ Static buffer: χωρίς malloc/free, χωρίς heap, word-aligned για RISC-V */
    static char print_buffer[PRINT_BUF_SIZE] __attribute__((aligned(4)));

    while (true)
    {
        if (xQueueReceive(s_csi_queue, &item, portMAX_DELAY) != pdTRUE)
            continue;

        /* ✅ Drop warning — atomic read + reset */
        unsigned int drops = atomic_exchange(&s_drop_count, 0);
        if (drops > 0)
        {
            ESP_LOGW(TAG, "WARNING: %u CSI frames dropped! (Queue Overflow)", drops);
        }

        /* Header — μία φορά */
        if (!s_header_printed)
        {
            ets_printf("type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,"
                       "channel,local_timestamp,sig_len,rx_state,len,first_word,data\n");
            s_header_printed = true;
        }

        /* ✅ Όλα σε ένα buffer — ένα μόνο ets_printf call */
        int size = PRINT_BUF_SIZE;
        int pos = snprintf(print_buffer, size,
                           "CSI_DATA,%lu," MACSTR ",%d,%d,%d,%d,%d,%d,%lu,%lu,%d,%d,%d,\"[",
                           (unsigned long)item.rx_id,
                           MAC2STR(item.mac),
                           (int)item.rx_ctrl.rssi,
                           (int)item.rx_ctrl.rate,
                           (int)item.rx_ctrl.noise_floor,
                           (int)item.fft_gain,
                           (int)item.agc_gain,
                           (int)item.rx_ctrl.channel,
                           (unsigned long)item.rx_ctrl.timestamp,
                           (unsigned long)item.rx_ctrl.sig_len,
                           (int)item.rx_ctrl.rx_state,
                           (int)item.len,
                           (int)item.first_word_invalid);

        /* Subcarriers */
        for (int i = 0; i < item.len; i++)
        {
            int written = snprintf(print_buffer + pos, size - pos,
                                   "%s%d",
                                   (i == 0) ? "" : ",",
                                   (int16_t)(item.compensate_gain * item.buf[i]));
            if (written > 0 && pos + written < size)
            {
                pos += written;
            }
            else
            {
                /* ✅ FIX: Buffer overflow — atomic count αντί για silent drop */
                atomic_fetch_add(&s_drop_count, 1);
                break;
            }
        }

        /* Κλείσιμο και εκτύπωση */
        if (pos + 4 < size)
        {
            snprintf(print_buffer + pos, size - pos, "]\"\n");
            ets_printf("%s", print_buffer);
        }
        else
        {
            /* ✅ FIX: Αν δεν χωράει το closing, μετράμε σαν drop */
            atomic_fetch_add(&s_drop_count, 1);
        }
    }
}

/* ════════════════════════════════════════════════════════════════════════
   6. CSI INIT
   ════════════════════════════════════════════════════════════════════════ */
static void wifi_csi_init(void)
{
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));

    wifi_csi_config_t csi_config = {
        .enable = true,
        .acquire_csi_legacy = false,
        .acquire_csi_ht20 = true,
        .acquire_csi_ht40 = true,
        .acquire_csi_su = true,
        .acquire_csi_mu = true,
        .acquire_csi_dcm = true,
        .acquire_csi_beamformed = true,
        .acquire_csi_he_stbc = 2,
        .val_scale_cfg = false,
        .dump_ack_en = false,
        .reserved = false};
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
}

/* ════════════════════════════════════════════════════════════════════════
   7. MAIN
   ════════════════════════════════════════════════════════════════════════ */
void app_main(void)
{
    /* ✅ NVS WEAR-OUT PROTECTION */
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND)
    {
        esp_reset_reason_t reason = esp_reset_reason();
        if (reason == ESP_RST_PANIC || reason == ESP_RST_WDT)
        {
            ESP_LOGE(TAG, "Crash loop detected! Halting to protect flash.");
            while (1)
            {
                vTaskDelay(pdMS_TO_TICKS(1000));
            }
        }
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    /* ✅ Queue με null check — restart αν αποτύχει */
    s_csi_queue = xQueueCreate(CSI_QUEUE_SIZE, sizeof(csi_queue_item_t));
    if (s_csi_queue == NULL)
    {
        ESP_LOGE(TAG, "FATAL: Queue creation failed! Free heap: %lu bytes",
                 (unsigned long)esp_get_free_heap_size());
        esp_restart();
    }

    /* ✅ Task με return check — restart αν αποτύχει */
    BaseType_t task_ret = xTaskCreate(csi_print_task, "csi_print",
                                      8192, NULL, 5, NULL);
    if (task_ret != pdPASS)
    {
        ESP_LOGE(TAG, "FATAL: Task creation failed! Free heap: %lu bytes",
                 (unsigned long)esp_get_free_heap_size());
        esp_restart();
    }

    /* ✅ Startup logs πριν τη σίγαση */
    antenna_init();
    wifi_init();

    esp_now_peer_info_t peer = {
        .channel = WIFI_CHANNEL,
        .ifidx = WIFI_IF_STA,
        .encrypt = false,
        .peer_addr = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff},
    };
    wifi_esp_now_init(peer);
    wifi_csi_init();

    /* ✅ WDT attach με έλεγχο */
    esp_err_t wdt_err = esp_task_wdt_add(NULL);
    if (wdt_err != ESP_OK)
    {
        ESP_LOGW(TAG, "WDT attach failed: %s", esp_err_to_name(wdt_err));
    }

    ESP_LOGI(TAG, "=========== CSI RECV READY ===========");
    ESP_LOGI(TAG, "Channel: %d | Sender MAC: " MACSTR,
             WIFI_CHANNEL, MAC2STR(KNOWN_SENDER_MAC));

    /* ✅ ΣΩΣΤΗ ΣΕΙΡΑ: Σίγαση logs ΜΕΤΑ τα startup messages
       Αν γίνει πριν, τα antenna/wifi init logs δεν φαίνονται.
       Αφήνουμε μόνο WARN ενεργό για τα drop warnings. */
    esp_log_level_set("*", ESP_LOG_NONE);
    esp_log_level_set(TAG, ESP_LOG_WARN);

    /* Keep-alive — τα δεδομένα βγαίνουν από το csi_print_task */
    while (true)
    {
        esp_task_wdt_reset();
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}