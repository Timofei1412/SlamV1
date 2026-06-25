'''
    Формат отпаравки:
    - Первая буква - тип пакета
    - массив данных с ";" в виде разделителя. если данные численные то формат всегда 3 цифры с ведущими нулями
    - Просто 0 (мусор) чтобы длинна всех пакетов была ровно N символов
'''
import serial
import logging
import threading
import time


class ESPCommunication: 
    def __init__(self, port:str = '/dev/serial0', packetLength:int = 25, baud:int = 115200, debug:bool = False):
        logging.info(f"Подключаемся: {port} | debug = {debug} | baud = {baud}")
        try:
            self.port = port
            self.packetLength = packetLength
            # timeout=0.1 позволяет неблокирующему чтению в потоке работать быстрее
            self.ser = serial.Serial(port, baud, timeout=0.1)
            self.debug = debug
            
            # Буфер для приема данных от ESP
            self.buffer = ""
            self.running = True
            self.virtualConnection = False
            # Запускаем поток чтения ответов от ESP
            self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self.read_thread.start()
        except Exception as e:
            print(f"ошибка {e}")
            self.packetLength = packetLength
            self.debug = debug
            self.virtualConnection = True
            logging.error(e)

        if self.virtualConnection:
            logging.warning(f"Подключение не выполнено, используем виртуальное подключение")
        else:
            logging.info(f"Готовы к работе")


    def makeSize(self, data:str, length:int):
        """
        Функция для подготовки данных к отправке
        Возвращает строку длины length, где сначала идет data, потом заполняющие "0"
        """
        return data + "0" * (length - len(data))

    def _read_loop(self):
        """Фоновый поток для чтения ответов от ESP"""
        while self.running:
            try:
                if self.ser.in_waiting > 0:
                    data = self.ser.read(self.ser.in_waiting).decode('utf-8', errors='ignore')
                    self._process_incoming(data)
            except Exception as e:
                logging.error(f"Read error: {e}")
            time.sleep(0.01)

    def _process_incoming(self, data: str):
        """Обработка входящих текстовых данных от ESP"""
        self.buffer += data
        # Если используем '!' как разделитель, раскомментируйте логику ниже
        # while '!' in self.buffer:
        #     packet, self.buffer = self.buffer.split('!', 1)
        #     if packet: logger.info(f"ESP: {packet}")
        
        # Пока просто выводим все, что пришло, если это не часть пакета команды
        logging.info(data.split())

        if self.debug and data.strip():
            print(f"ESP <- {data.strip()}")

    def sendMotionCommand(self, linearSpeed, rotationalSpeed):
        # Формируем полезную нагрузку
        packet = f"L{linearSpeed:03d};{rotationalSpeed:03d}"
        # Дополняем нулями до нужной длины
        full_packet = self.makeSize(packet, self.packetLength)
        
        logging.info(f"Sent Motion: {full_packet}")
        
        if self.debug:
            print(f"RPi -> ESP: {full_packet}")
        
        if not self.virtualConnection:
            self.ser.write(full_packet.encode())

    def sendMode(self, mode):
        # Формат MXXX...
        packet = f"M{mode:03d}"
        full_packet = self.makeSize(packet, self.packetLength)
        
        logging.info(f"Sent Mode: {full_packet}")
        if self.debug:
            print(f"RPi -> ESP: {full_packet}")
        
        
        if not self.virtualConnection:
            self.ser.write(full_packet.encode())

    def close(self):
        self.running = False
        if not self.virtualConnection:
            self.ser.close()

if __name__ == "__main__":
    esp = ESPCommunication(debug=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        esp.close()