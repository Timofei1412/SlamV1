#include <Arduino.h>

#define RPI_SERIAL Serial1
#define BAUD_RATE 115200

#define TX_PACKET_SIZE 18       // M(1) + mode(1) + 4*int32(16)
#define RX_PACKET_SIZE 17       // M(1) + 4*int16(8) + 4*uint16(8)
#define TEXT_MAX_LENGTH 64      // Максимальная длина текстового сообщения

#pragma pack(push, 1)
struct RxPacket {
    uint8_t header;      // 'M'
    int16_t speeds[4];
    uint16_t servos[4];
};

struct TxPacket {
    uint8_t header;     // 'M'
    uint8_t mode;
    int32_t values[4];
};
#pragma pack(pop)

// Глобальные переменные состояния
uint8_t currentMode = 0;
int16_t motorSpeeds[4] = {0};
uint16_t servoPositions[4] = {0};

// Буфер для приема команд
static uint8_t rxBuffer[RX_PACKET_SIZE];
static uint8_t rxIndex = 0;
static bool rxSync = false;

void setupComms() {
    RPI_SERIAL.begin(BAUD_RATE);
    delay(50);

    // Отправляем начальный пакет
    TxPacket initPkt = {'M', 0, {0, 0, 0, 0}};
    RPI_SERIAL.write((uint8_t*)&initPkt, sizeof(TxPacket));
}

void readComms() {
    while (RPI_SERIAL.available() > 0) {
        uint8_t b = RPI_SERIAL.read();

        if (!rxSync) {
            // Ожидаем заголовок 'M'
            if (b == 'M') {
                rxBuffer[0] = b;
                rxIndex = 1;
                rxSync = true;
            }
            // Игнорируем другие байты
            continue;
        }

        rxBuffer[rxIndex++] = b;

        if (rxIndex >= RX_PACKET_SIZE) {
            // Полный пакет получен
            RxPacket* pkt = (RxPacket*)rxBuffer;
            
            // Копируем данные
            for (int i = 0; i < 4; i++) {
                motorSpeeds[i] = pkt->speeds[i];
                servoPositions[i] = pkt->servos[i];
            }
            
            // Если speeds[0] >= 100, это команда смены режима
            if (motorSpeeds[0] >= 100) {
                currentMode = (uint8_t)motorSpeeds[0];
            }
            
            rxSync = false;
            rxIndex = 0;
        }
    }
}

void sendTelemetry(int32_t val1, int32_t val2, int32_t val3, int32_t val4) {
    TxPacket pkt;
    pkt.header = 'M';
    pkt.mode = currentMode;
    pkt.values[0] = val1;
    pkt.values[1] = val2;
    pkt.values[2] = val3;
    pkt.values[3] = val4;
    RPI_SERIAL.write((uint8_t*)&pkt, sizeof(TxPacket));
}

void sendText(const char* format, ...) {
    // Формируем текстовое сообщение
    char buffer[TEXT_MAX_LENGTH];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, TEXT_MAX_LENGTH, format, args);
    va_end(args);
    
    // Отправляем: 'T' + длина + текст
    uint8_t len = strlen(buffer);
    RPI_SERIAL.write('T');
    RPI_SERIAL.write(len);
    RPI_SERIAL.write((const uint8_t*)buffer, len);
}

// Удобные макросы для отладки
#define LOG_INFO(msg) sendText("[INFO] %s", msg)
#define LOG_WARN(msg) sendText("[WARN] %s", msg)
#define LOG_ERROR(msg) sendText("[ERROR] %s", msg)
#define LOG_DEBUG(msg) sendText("[DEBUG] %s", msg)