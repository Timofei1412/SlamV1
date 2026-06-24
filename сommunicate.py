'''
    Формат отпаравки:
    - Первая буква - тип пакета
    - массив данных с ";" в виде разделителя. если данные численные то формат всегда 3 цифры с ведущими нулями
    - Последняя буква аналогична первой
    - Просто 0 (мусор) чтобы длинна всех пакетов была ровно N символов
'''
import serial
import logging

class ESPCommunication: 
    def __init__(self, port:str, packetLength:int = 25, baud:int = 115200, debug:bool = False):
        self.port = port
        self.ser = serial.Serial(port, baud, timeout=1)
        self.debug = debug
        self.packetLength = packetLength

    def makeSize(data:str, length:int):
        """
        Функция для подготовки данных к отправке
        Возращяет строку длины length, где сначала идет data, потом заполняющие "0"
        """
        return data + "".join(["0" for i in range(length -len(data))])

    def sendMotionCommand(self, linearSpeed, rotationalSpeed):
        packet = f"L{linearSpeed};{rotationalSpeed}L"
        logging.info(packet)
        if self.debug:
            print(packet)
        packet = self.makeSize(packet, self.packetLength)
        self.ser.write(packet.encode())

    def sendServoCommand(self, servoArm:bool, servoClaw:bool):
        packet = f"S{servoArm:03d};{servoClaw:03d}S"    
        if self.debug:
            print(packet)
        logging.info(packet)
        packet = self.makeSize(packet, self.packetLength)
        self.ser.write(packet.encode())

    def sendMode(self, mode):
        raise NotImplementedError
    

if __name__ == "__main__":
    pass