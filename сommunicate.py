import serial
import struct
import logging
import threading
import time
from typing import List, Optional, Dict, Any
from datetime import datetime

class ESPCommunication:
    # TX: 'M'(1) + 4*int16(8) + 4*uint16(8) = 17 байт
    TX_STRUCT = struct.Struct('<BB4b4b')
    # RX: 'M'(1) + uint8(1) + 4*int32(16) = 18 байт
    RX_STRUCT = struct.Struct('<Bi4i')
    RX_HEADER = b'M'
    RX_PACKET_SIZE = 15

    def __init__(self, port: str = '/dev/ttyAMA0', baud: int = 115200, debug: bool = False):
        self.debug = debug
        self.port = port
        self.baud = baud

        self.virtualConnection = False
        self.ser: Optional[serial.Serial] = None
        
        self.running = True
        # Хранилище последнего пакета
        self._last_packet = [0]
        self._packet_lock = threading.Lock()

        try:
            self.ser = serial.Serial(port, baud, timeout=0.1)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            logging.info(f"ESP Connected: {port} @ {baud}")
        except Exception as e:
            logging.error(f"Serial init failed: {e}. Virtual mode.")
            self.virtualConnection = True

        self.read_thread = threading.Thread(target=self._read_loop)
        self.read_thread.start()

    def _read_loop(self):
        """Неблокирующий цикл чтения с точной сборкой пакетов"""
        while self.running:
            if self.virtualConnection or not self.ser:
                return

            try:
                if self.ser.in_waiting > 7:
                    data = self.ser.read()
                    # logging.warning("started")
                    if data == b"M":
                        telem = []
                        data = self.ser.read()

                        while(data != b"M"):
                            # logging.info(int(data.hex(), 16))
                            telem.append(int(data.hex(), 16))
                            data = self.ser.read()
                        self._last_packet = telem
                        # logging.exception(telem)
            except Exception as e:
                logging.error(f"RX Loop Error: {e}")



    def get_last_packet(self) -> Dict[str, Any]:
        """Мгновенное получение копии последнего валидного пакета от ESP"""
        with self._packet_lock:
            return self._last_packet.copy()

    def sendMotionCommand(self, mode: int,  speeds: List[int], servos: List[int]):
        """
        Отправка команды движения
        :param mode int
        :param speeds: список из 4-х int8 (-32768..32767)
        :param servos: список из 4-х uint8 (0..65535)
        """
        if len(speeds) != 4 or len(servos) != 4:
            logging.error("Speeds and Servos must have exactly 4 elements each")
            return
        
        try:
            # packet = self.TX_STRUCT.pack(ord('M'), mode, *speeds, *servos)
            data = [ord("M"), mode]
            for i in speeds:
                data.append(i + 100)
            for i in servos:
                data.append(i)

            packet = bytes(data)
            logging.info(f"TX -> Mode: {mode} Speeds:{speeds} Servos:{servos}")

            if self.debug:
                print(f"RPi -> ESP: {packet.hex(bytes_per_sep=2)}")

            if not self.virtualConnection and self.ser:
                self.ser.write(packet)

        except struct.error as e:
            logging.error(f"Pack error: {e}")

    def close(self):
        self.running = False
        if self.ser and not self.virtualConnection:
            self.ser.close()
        logging.info("ESP Communication closed")


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    timestamp = datetime.now().strftime('%d_%H-%M-%S')
    log_file = f'Output/Logs/app_{timestamp}.log'

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        # datefmt='%H:%M:%S.%f',
        format='%(asctime)s | %(levelname)-8s | %(filename)s | %(message)s',
        level=logging.INFO,
        encoding='utf-8',
        # backupCount=3,
        )
    esp = ESPCommunication()
    a = time.time()
    while esp.get_last_packet() != [1]:
        esp.sendMotionCommand(0, [0, 0, 0, 0], [0, 0, 0, 0])
        print((time.time() - a)*1000)
        a = time.time()
        # time.sleep(.1)
    for i in range(100):
        esp.sendMotionCommand(0, [10, 0, 0, 0], [0, 0, 0, 0])
        time.sleep(.1)
    esp.sendMotionCommand(0, [0, 0, 0, 0], [0, 0, 0, 0])
    # print(esp.get_last_packet())
    esp.close()


# 4d000000000500000000000000000000000000000000000000000000000000000000000000
# 4d00|0005|0000|0000|0000|0000|0000|0000|0000|00