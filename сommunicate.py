import serial
from tools import makeSize
import logging

class ESPCommunication:
    '''
    формат отпаравки:
    - Первая буква - тип пакета
    - массив данных с ";" в виде разделителя. если данные численные то формат всегда 3 цифры с ведущими нулями
    - Последняя буква аналогична первой
    - Просто 0 (мусор) чтобы длинна всех пакетов была ровно N символов
    '''
    
    def __init__(self, port:str, packetLength:int = 25, baud:int = 115200, debug:bool = False):
        self.port = port
        self.ser = serial.Serial(port, baud, timeout=1)
        self.debug = debug
        self.packetLength = packetLength
            
    def sendLineData(self, data:list[int]):
        packet = f"L{data[0]:03d};{data[1]:03d};{data[2]:03d};{data[3]:03d}L"
        logging.info(packet)
        if self.debug:
            print(packet)
        packet = makeSize(packet, self.packetLength)
        self.ser.write(packet.encode())

    def sendServoCommand(self, servoArm:bool, servoClaw:bool):
        packet = f"S{servoArm:03d};{servoClaw:03d}S"    
        if self.debug:
            print(packet)
        logging.info(packet)
        packet = makeSize(packet, self.packetLength)
        self.ser.write(packet.encode())
    
    def sendMode(self, mode:str):
        packet = f"M{mode}M"
        raise NotImplementedError

if __name__ == "__main__":
    pass