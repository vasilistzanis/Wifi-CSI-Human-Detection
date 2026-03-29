/*
 * ESP32-C6 Production Sender (Thesis Grade) — Final
 * Features: Magic Header Payload, Deterministic Timing, TX Power Lock,
 * External Antenna, NVS Wear-out Protection, HT40 Wi-Fi 4 Init Fix,
 * WDT Check, Timing Reset after reinit, band_mode error check
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#include "esp_system.h"
#include "nvs_flash.h"
#include "esp_mac.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_now.h"
#include "esp_timer.h"
#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_task_wdt.h"

/* ── Ρυθμίσεις ──────────────────────────────────────────────────────────── */
#define WIFI_CHANNEL 11
#define CONFIG_SEND_FREQUENCY 100
#define SEND_PERIOD_US (1000000 / CONFIG_SEND_FREQUENCY)
#define TX_POWER_FIXED 72 // 18dBm — 0.25dBm units
#define WIFI_ENABLE GPIO_NUM_3
#define WIFI_ANT_CONFIG GPIO_NUM_14

static const uint8_t SENDER_MAC[] = {0x1a, 0x00, 0x00, 0x00, 0x00, 0x00};
static const uint8_t TARGET_RECEIVER_MAC[] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
static const char *TAG = "csi_send";

#define RADAR_MAGIC_SIGNATURE 0xA1B2C3D4

typedef struct __attribute__((packed))
{
    uint32_t magic;
    uint32_t sequence_id;
} radar_payload_t;

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
    ESP_ERROR_CHECK(esp_wifi_set_mac(WIFI_IF_STA, SENDER_MAC));

    /* ✅ TX Power lock + verification */
    ESP_ERROR_CHECK(esp_wifi_set_max_tx_power(TX_POWER_FIXED));
    int8_t actual_power = 0;
    esp_wifi_get_max_tx_power(&actual_power);
    ESP_LOGI(TAG, "TX power locked at %.2f dBm", actual_power / 4.0f);
}

/* ════════════════════════════════════════════════════════════════════════
   3. ESP-NOW SEND CALLBACK — Silent (100Hz, UART bottleneck prevention)
   ════════════════════════════════════════════════════════════════════════ */
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 0, 0)
static void esp_now_send_cb(const esp_now_send_info_t *send_info,
                            esp_now_send_status_t status)
#else
static void esp_now_send_cb(const uint8_t *mac, esp_now_send_status_t status)
#endif
{
    /* Σκόπιμα κενό — έλεγχος γίνεται μέσω fail_count στο loop */
}

/* ════════════════════════════════════════════════════════════════════════
   4. ESP-NOW INIT
   ════════════════════════════════════════════════════════════════════════ */
static void wifi_esp_now_init(esp_now_peer_info_t peer)
{
    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_register_send_cb(esp_now_send_cb));
    ESP_ERROR_CHECK(esp_now_set_pmk((uint8_t *)"pmk1234567890123"));
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));

    esp_now_rate_config_t rate_config = {
        .phymode = WIFI_PHY_MODE_HT40,
        .rate = WIFI_PHY_RATE_MCS0_LGI,
        .ersu = false,
        .dcm = false};
    ESP_ERROR_CHECK(esp_now_set_peer_rate_config(peer.peer_addr, &rate_config));
}

/* ════════════════════════════════════════════════════════════════════════
   5. MAIN
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

    antenna_init();
    wifi_init();

    esp_now_peer_info_t peer = {
        .channel = WIFI_CHANNEL,
        .ifidx = WIFI_IF_STA,
        .encrypt = false,
    };
    memcpy(peer.peer_addr, TARGET_RECEIVER_MAC, 6);
    wifi_esp_now_init(peer);

    /* ✅ WDT attach με έλεγχο */
    esp_err_t wdt_err = esp_task_wdt_add(NULL);
    if (wdt_err != ESP_OK)
    {
        ESP_LOGW(TAG, "WDT attach failed: %s", esp_err_to_name(wdt_err));
    }
    else
    {
        ESP_LOGI(TAG, "Watchdog attached to main task");
    }

    ESP_LOGI(TAG, "=========== CSI SEND ===========");
    ESP_LOGI(TAG, "Channel: %d | Freq: %d Hz | MAC: " MACSTR,
             WIFI_CHANNEL, CONFIG_SEND_FREQUENCY, MAC2STR(SENDER_MAC));

    uint32_t fail_count = 0;
    uint32_t count = 0;
    int64_t next_send_us = esp_timer_get_time();

    radar_payload_t payload = {
        .magic = RADAR_MAGIC_SIGNATURE,
        .sequence_id = 0};

    /* ✅ DETERMINISTIC TIMING LOOP */
    while (true)
    {
        esp_task_wdt_reset();

        payload.sequence_id = count;
        ret = esp_now_send(peer.peer_addr,
                           (const uint8_t *)&payload,
                           sizeof(payload));

        if (ret != ESP_OK)
        {
            fail_count++;
            if (fail_count > 20)
            {
                ESP_LOGE(TAG, "Too many errors [%lu] — reinit ESP-NOW: %s",
                         (unsigned long)fail_count, esp_err_to_name(ret));
                esp_now_deinit();
                vTaskDelay(pdMS_TO_TICKS(500));
                wifi_esp_now_init(peer);
                fail_count = 0;
                /* ✅ FIX: Reset timing μετά από reinit
                   Χωρίς αυτό, το loop θα προσπαθούσε να στείλει
                   ~50 frames αμέσως για να "αναπληρώσει" το χαμένο χρόνο */
                next_send_us = esp_timer_get_time();
            }
        }
        else
        {
            fail_count = 0;
            count++;
        }

        /* Deterministic timing — αφαιρεί τον χρόνο εκτέλεσης */
        next_send_us += SEND_PERIOD_US;
        int64_t now = esp_timer_get_time();
        int64_t sleep_us = next_send_us - now;

        if (sleep_us > 0)
        {
            usleep((useconds_t)sleep_us);
        }
        else
        {
            /* Drift — reset timer */
            next_send_us = esp_timer_get_time();
        }
    }
}