#include <Arduino.h>

#define RPI_SERIAL Serial1
#define BAUD_RATE 115200

#define TX_PACKET_SIZE 18  // M(1) + mode(1) + 4*int32(16)
#define RX_PACKET_SIZE 18  // M(1) + mode(1) + 4*int16(8) + 4*uint16(8)

#define MOTOR_SPEED_LIMIT 100  // ±100

#pragma pack(push, 1)
struct RxPacket {
    uint8_t header;     // 'M'
    uint8_t mode;
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
uint8_t mode = 0;
int16_t motorSpeeds[4] = {0};
uint16_t servoPositions[4] = {0};

static uint8_t rxBuffer[RX_PACKET_SIZE];
static uint8_t rxIndex = 0;
static bool rxSync = false;



void setupComms() {
    RPI_SERIAL.begin(BAUD_RATE);
    delay(50);

    TxPacket initPkt = {'M', 0, {0, 0, 0, 0}};
    RPI_SERIAL.write((uint8_t*)&initPkt, sizeof(TxPacket));
}

void readComms() {
    while (RPI_SERIAL.available() > 0) {
        uint8_t b = RPI_SERIAL.read();
        if (b == 'M') {
            rxBuffer[0] = b;
            for (rxBuffer = 1; rxBuffer < RX_PACKET_SIZE; rxBuffer += 1){
                uint8_t b = RPI_SERIAL.read();
                rxBuffer[rxIndex++] = b;
            }
            RxPacket* pkt = (RxPacket*)rxBuffer;
            
            // Обновляем состояние
            mode = pkt->mode;
            
            // Ограничиваем скорости двигателей ±100
            for (int i = 0; i < 4; i++) {
                motorSpeeds[i] = pkt->speeds[i] - 100;
            }            
            rxIndex = 0;
        }
    }
}

void sendTelemetry(uint8_t currentMode, int32_t val1, int32_t val2, int32_t val3, int32_t val4) {
    TxPacket pkt;
    pkt.header = 'M';
    pkt.mode = currentMode;
    pkt.values[0] = val1;
    pkt.values[1] = val2;
    pkt.values[2] = val3;
    pkt.values[3] = val4;
    RPI_SERIAL.write((uint8_t*)&pkt, sizeof(TxPacket));
}