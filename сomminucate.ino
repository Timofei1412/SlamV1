#include <Arduino.h>

#define RPI_SERIAL Serial1
#define PACKET_LENGTH 25
#define BAUD_RATE 115200

String buffer = "";
extern uint8_t mode;
extern int16_t linSpeed;
extern int16_t rotSpeed;

// mode 0 - wait
// mode 1 - followLine
// mode 2 - take
// mode 3 - put

void setupComms() {
    RPI_SERIAL.begin(BAUD_RATE);
    // Небольшая задержка для стабилизации соединения
    delay(100); 
    sendToPi("INIT_OK");
}

// Неблокирующая функция чтения. Вызывать в loop()
void readComms() {
    while (RPI_SERIAL.available() > 0) {
        char c = RPI_SERIAL.read();
        buffer += c;
        
        // Как только набрали нужную длину пакета - обрабатываем
        if (buffer.length() >= PACKET_LENGTH) {
            processPacket(buffer.substring(0, PACKET_LENGTH));
            buffer = "";
        }
    }
}

// Функция отправки любого текста на Raspberry Pi
void sendToPi(String text) {
    RPI_SERIAL.println(text);
}

void processPacket(String packet) {
    // Проверяем первую и последнюю букву (валидация формата)
    char type = packet[0];
    char lastChar = packet[PACKET_LENGTH - 1];
    
    if (type == 'L') {
        // Формат: LXXX;YYY00000000000000000
        // Ищем разделитель
        int sep = packet.indexOf(';');
        if (sep != -1) {
            // Извлекаем подстроки между L и ; а также между ; и концом полезных данных
            String linearStr = packet.substring(1, sep);

            
            String rotationalStr = packet.substring(sep + 1);
            
            linSpeed = linearStr.toInt();
            rotSpeed = rotationalStr.toInt();
            
            // Отправляем подтверждение на RPi
            sendToPi("MOTION_SET:" + String(linSpeed) + "," + String(rotSpeed));
        }
        
    } else if (type == 'M') {
        // Формат: MXXX000000000000000000000
        String dataStr = packet.substring(1);
        mode = dataStr.toInt();
        
        sendToPi("MODE_SET:" + String(mode));
    }
}