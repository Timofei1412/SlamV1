import serial
import struct
import logging
import threading
import time
from typing import List, Optional, Dict, Any


class ESPCommunication:
    # TX: 'M'(1) + 4*int16(8) + 4*uint16(8) = 17 байт
    TX_STRUCT = struct.Struct('<B4h4H')
    # RX: 'M'(1) + uint8(1) + 4*int32(16) = 18 байт
    RX_STRUCT = struct.Struct('<Bi4i')
    RX_HEADER = b'M'
    RX_PACKET_SIZE = 18

    def __init__(self, port: str = '/dev/serial0', baud: int = 115200, debug: bool = False):
        self.debug = debug
        self.port = port
        self.baud = baud

        self.virtualConnection = False
        self.ser: Optional[serial.Serial] = None
        self.running = True
        self._rx_buffer = bytearray()

        # Хранилище последнего пакета
        self._last_packet: Dict[str, Any] = {
            'mode': 0,
            'values': [0, 0, 0, 0],
            'timestamp': 0.0,
            'valid': False
        }
        self._packet_lock = threading.Lock()

        try:
            self.ser = serial.Serial(port, baud, timeout=0.1)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            logging.info(f"ESP Connected: {port} @ {baud}")
        except Exception as e:
            logging.error(f"Serial init failed: {e}. Virtual mode.")
            self.virtualConnection = True

        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.read_thread.start()

    def _read_loop(self):
        """Неблокирующий цикл чтения с точной сборкой пакетов"""
        while self.running:
            if self.virtualConnection or not self.ser:
                continue

            try:
                waiting = self.ser.in_waiting
                if waiting > 0:
                    data = self.ser.read(waiting)
                    self._rx_buffer.extend(data)

                    while len(self._rx_buffer) >= self.RX_PACKET_SIZE:
                        idx = self._rx_buffer.find(self.RX_HEADER)
                        if idx == -1:
                            self._rx_buffer.clear()
                            break

                        if idx > 0:
                            logging.warning(f"Discarding {idx} garbage bytes")
                            del self._rx_buffer[:idx]

                        if len(self._rx_buffer) < self.RX_PACKET_SIZE:
                            break

                        packet_bytes = bytes(self._rx_buffer[:self.RX_PACKET_SIZE])
                        del self._rx_buffer[:self.RX_PACKET_SIZE]
                        self._handle_rx_packet(packet_bytes)

            except Exception as e:
                logging.error(f"RX Loop Error: {e}")

    def _handle_rx_packet(self, raw: bytes):
        """Парсинг и сохранение последнего пакета"""
        try:
            _, mode, v1, v2, v3, v4 = self.RX_STRUCT.unpack(raw)

            with self._packet_lock:
                self._last_packet['mode'] = mode
                self._last_packet['values'] = [v1, v2, v3, v4]
                self._last_packet['timestamp'] = time.time()
                self._last_packet['valid'] = True

            logging.info(f"RX <- Mode:{mode} Data:[{v1}, {v2}, {v3}, {v4}]")
            if self.debug:
                print(f"ESP <- M{mode} | {v1} {v2} {v3} {v4}")
        except struct.error as e:
            logging.error(f"Unpack error: {e} | Raw: {raw.hex()}")

    def get_last_packet(self) -> Dict[str, Any]:
        """Мгновенное получение копии последнего валидного пакета от ESP"""
        with self._packet_lock:
            return self._last_packet.copy()

    def sendMotionCommand(self, speeds: List[int], servos: List[int]):
        """
        Отправка команды движения
        :param speeds: список из 4-х int16 (-32768..32767)
        :param servos: список из 4-х uint16 (0..65535)
        """
        if len(speeds) != 4 or len(servos) != 4:
            logging.error("Speeds and Servos must have exactly 4 elements each")
            return

        try:
            packet = self.TX_STRUCT.pack(ord('M'), *speeds, *servos)
            logging.info(f"TX -> Speeds:{speeds} Servos:{servos}")

            if self.debug:
                print(f"RPi -> ESP: {packet.hex()}")

            if not self.virtualConnection and self.ser:
                self.ser.write(packet)

        except struct.error as e:
            logging.error(f"Pack error: {e}")

    def sendMode(self, mode: int):
        """
        Отправка команды режима.
        Режим передается как第一个 speeds[0].
        
        :param mode: Режим работы (int16)
        """
        # Отправляем как команду движения с нулевыми скоростями и серво
        zeros = [0, 0, 0, 0]
        mode_speed = [mode, 0, 0, 0]
        self.sendMotionCommand(mode_speed, zeros)
        logging.info(f"TX -> Mode command sent: {mode}")

    def close(self):
        self.running = False
        if self.ser and not self.virtualConnection:
            self.ser.close()
        logging.info("ESP Communication closed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    esp = ESPCommunication(debug=True)
    try:
        while True:
            state = esp.get_last_packet()
            if state['valid']:
                age_ms = (time.time() - state['timestamp']) * 1000
                print(f"Mode: {state['mode']} | Vals: {state['values']} | Age: {age_ms:.1f}ms")
            time.sleep(0.1)
    except KeyboardInterrupt:
        esp.close()