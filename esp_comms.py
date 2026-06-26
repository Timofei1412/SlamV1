import serial
import struct
import logging
import threading
import time
import os
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple


# =============================================================================
# DEBUG FLAG
# =============================================================================
DEBUG = True  # Set to False to disable debug output


def log_debug(text: str):
    """Вывести debug текст если DEBUG=True."""
    if DEBUG:
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        prefix = f"[ESP_DEBUG][{timestamp}]"
        print(f"{prefix} {text}")
        logging.debug(f"{prefix} {text}")


def log_esp_rx(data: dict):
    """Логировать принятые данные от ESP."""
    log_debug(f"RX <- mode={data.get('mode')}, values={data.get('values')}")


def log_esp_tx(speeds: List[int], servos: List[int]):
    """Логировать отправленные данные ESP."""
    log_debug(f"TX -> speeds={speeds}, servos={servos}")


class ESPCommunication:
    """
    Класс для коммуникации с ESP32.
    
    Протокол:
    - RPi -> ESP: 'M'(1) + 4×int16(8) + 4×uint16(8) = 17 байт
    - ESP -> RPi: 'M'(1) + uint8(1) + 4×int32(16) = 18 байт
    - ESP -> RPi (текст): 'T'(1) + uint8(len) + data(n) = 2+n байт
    
    Каждой команде от RPi должен соответствовать ответ от ESP.
    """
    
    # TX: 'M'(1) + 4×int16(8) + 4×uint16(8) = 17 байт
    TX_STRUCT = struct.Struct('<B4h4H')
    # RX: 'M'(1) + uint8(1) + 4×int32(16) = 18 байт
    RX_STRUCT = struct.Struct('<BB4i')
    RX_PACKET_SIZE = 18
    
    # Текстовый пакет: 'T'(1) + uint8(len) + data(n)
    TEXT_HEADER = b'T'
    
    # Таймауты и лимиты
    RESPONSE_TIMEOUT_MS = 500  # Таймаут ожидания ответа от ESP
    MAX_MISSED_RESPONSES = 10  # Максимум пропущенных ответов перед выходом

    def __init__(self, port: str = '/dev/serial0', baud: int = 115200, debug: bool = False):
        self.debug = debug
        self.port = port
        self.baud = baud

        self.virtualConnection = False
        self.ser: Optional[serial.Serial] = None
        self.running = True
        self._rx_buffer = bytearray()
        
        # Счетчики для мониторинга связи
        self._sent_count = 0
        self._received_count = 0
        self._missed_responses = 0
        self._last_command = ""
        
        # Синхронизация ответа
        self._response_event = threading.Event()
        self._last_response: Dict[str, Any] = {
            'mode': 0,
            'values': [0, 0, 0, 0],
            'valid': False
        }
        self._response_lock = threading.Lock()

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
                time.sleep(0.01)
                continue

            try:
                waiting = self.ser.in_waiting
                if waiting > 0:
                    data = self.ser.read(waiting)
                    self._rx_buffer.extend(data)
                    log_debug(f"RX buffer +{len(data)} bytes, total={len(self._rx_buffer)}")
                    
                    # Обработка буфера
                    self._process_rx_buffer()

            except Exception as e:
                logging.error(f"RX Loop Error: {e}")
                log_debug(f"RX Error: {e}")
                time.sleep(0.01)
    
    def _process_rx_buffer(self):
        """Обработка принятых данных из буфера"""
        while len(self._rx_buffer) > 0:
            # Проверяем на текстовый пакет
            if self._rx_buffer[0:1] == self.TEXT_HEADER:
                if len(self._rx_buffer) >= 2:
                    text_len = self._rx_buffer[1]
                    if len(self._rx_buffer) >= 2 + text_len:
                        text_data = bytes(self._rx_buffer[2:2 + text_len])
                        del self._rx_buffer[:2 + text_len]
                        self._handle_text_packet(text_data)
                    else:
                        break  # Ждем остаток текста
                else:
                    break  # Ждем байт длины
            elif self._rx_buffer[0:1] == b'M':
                if len(self._rx_buffer) >= self.RX_PACKET_SIZE:
                    packet_bytes = bytes(self._rx_buffer[:self.RX_PACKET_SIZE])
                    del self._rx_buffer[:self.RX_PACKET_SIZE]
                    self._handle_data_packet(packet_bytes)
                else:
                    break  # Ждем остаток пакета
            else:
                # Неизвестный байт - пропускаем
                unknown_byte = hex(self._rx_buffer[0])
                logging.warning(f"Discarding unknown byte: {unknown_byte}")
                log_debug(f"Unknown byte discarded: {unknown_byte}")
                del self._rx_buffer[:1]
    
    def _handle_text_packet(self, data: bytes):
        """Обработка текстового пакета от ESP"""
        try:
            text = data.decode('utf-8').rstrip('\x00')
            if text:
                logging.info(f"ESP TEXT: {text}")
                log_debug(f"ESP text: {text}")
                if self.debug:
                    print(f"[ESP] {text}")
        except Exception as e:
            logging.warning(f"Failed to decode text from ESP: {e}")
            log_debug(f"Text decode error: {e}")
    
    def _handle_data_packet(self, raw: bytes):
        """Обработка пакета данных от ESP"""
        try:
            _, mode, v1, v2, v3, v4 = self.RX_STRUCT.unpack(raw)

            with self._response_lock:
                self._last_response['mode'] = mode
                self._last_response['values'] = [v1, v2, v3, v4]
                self._last_response['valid'] = True

            # Сигнализируем о получении ответа
            self._response_event.set()
            self._received_count += 1
            
            log_esp_rx({'mode': mode, 'values': [v1, v2, v3, v4]})
            
            if self.debug:
                print(f"ESP -> RPi: Mode={mode} Data=[{v1}, {v2}, {v3}, {v4}]")
                
        except struct.error as e:
            logging.error(f"Unpack error: {e} | Raw: {raw.hex()}")
            log_debug(f"Unpack error: {e}")
    
    def _clear_response(self):
        """Очистка флага ответа перед отправкой команды"""
        self._response_event.clear()
        with self._response_lock:
            self._last_response['valid'] = False
    
    def send_command(self, speeds: List[int], servos: List[int], 
                     wait_response: bool = True) -> Tuple[bool, Dict[str, Any]]:
        """
        Отправка команды движения с ожиданием ответа.
        
        :param speeds: список из 4-х int16 (-32768..32767)
        :param servos: список из 4-х uint16 (0..65535)
        :param wait_response: ждать ли ответ от ESP
        
        :returns: Tuple[успех, данные_ответа]
        """
        if len(speeds) != 4 or len(servos) != 4:
            logging.error("Speeds and Servos must have exactly 4 elements each")
            log_debug(f"Invalid command: speeds/servos length error")
            return False, {}
        
        if self.virtualConnection:
            log_debug(f"VIRTUAL MODE: would send speeds={speeds}")
            return True, {'mode': 0, 'values': [0, 0, 0, 0], 'valid': True}
        
        try:
            # Формируем пакет
            packet = self.TX_STRUCT.pack(ord('M'), *speeds, *servos)
            command_str = f"M[{speeds}] [{servos}]"
            self._last_command = command_str
            
            # Очищаем флаг ответа
            self._clear_response()
            
            # Отправляем
            self.ser.write(packet)
            self._sent_count += 1
            
            log_esp_tx(speeds, servos)
            logging.info(f"TX -> speeds={speeds}, servos={servos}")
            
            if self.debug:
                print(f"RPi -> ESP: {packet.hex()}")
            
            if not wait_response:
                return True, {}
            
            # Ждем ответ с таймаутом
            if self._response_event.wait(timeout=self.RESPONSE_TIMEOUT_MS / 1000.0):
                # Получили ответ
                with self._response_lock:
                    self._missed_responses = 0  # Сброс счетчика пропущенных
                    return self._last_response['valid'], self._last_response.copy()
            else:
                # Таймаут
                self._missed_responses += 1
                logging.warning(f"ESP did not respond to: {command_str} "
                             f"(missed: {self._missed_responses}/{self.MAX_MISSED_RESPONSES})")
                log_debug(f"TIMEOUT: no response (missed={self._missed_responses})")
                
                if self._missed_responses >= self.MAX_MISSED_RESPONSES:
                    logging.error(f"Too many missed ESP responses ({self._missed_responses}). "
                                f"Connection lost. Exiting.")
                    log_debug(f"CONNECTION LOST: max missed responses reached")
                    self.running = False
                    return False, {}
                
                return False, {}
                
        except serial.SerialException as e:
            logging.error(f"Serial error: {e}")
            log_debug(f"Serial exception: {e}")
            return False, {}
        except struct.error as e:
            logging.error(f"Pack error: {e}")
            log_debug(f"Pack error: {e}")
            return False, {}

    def sendMotionCommand(self, speeds: List[int], servos: List[int]) -> Tuple[bool, Dict[str, Any]]:
        """
        Отправка команды движения с ожиданием ответа.
        Алиас для send_command для совместимости.
        
        :param speeds: список из 4-х int16 (-32768..32767)
        :param servos: список из 4-х uint16 (0..65535)
        
        :returns: Tuple[успех, данные_ответа]
        """
        return self.send_command(speeds, servos, wait_response=True)

    def sendMode(self, mode: int) -> Tuple[bool, Dict[str, Any]]:
        """
        Отправка команды режима.
        Режим передается как mode в пакете.
        
        :param mode: Режим работы (0-255)
        
        :returns: Tuple[успех, данные_ответа]
        """
        # Отправляем команду с mode как первым элементом values
        # В ESP это будет интерпретироваться как команда смены режима
        speeds = [mode, 0, 0, 0]  # mode в speeds[0]
        servos = [0, 0, 0, 0]
        
        logging.info(f"TX -> Mode command: {mode}")
        return self.send_command(speeds, servos, wait_response=True)

    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику коммуникации."""
        return {
            'sent': self._sent_count,
            'received': self._received_count,
            'missed': self._missed_responses,
            'success_rate': (self._received_count / self._sent_count * 100) 
                           if self._sent_count > 0 else 0,
            'virtual': self.virtualConnection
        }
    
    def is_connected(self) -> bool:
        """Проверить статус соединения."""
        return self.running and not (self._missed_responses >= self.MAX_MISSED_RESPONSES)

    def close(self):
        self.running = False
        if self.ser and not self.virtualConnection:
            self.ser.close()
        
        stats = self.get_stats()
        logging.info(f"ESP Communication closed. Stats: {stats}")
        print(f"\nCommunication stats: sent={stats['sent']}, "
              f"received={stats['received']}, missed={stats['missed']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
    
    esp = ESPCommunication(debug=True)
    try:
        print("\n=== ESP Communication Test ===")
        print("Commands: [speeds], [servos]")
        print("Press Ctrl+C to exit\n")
        
        while esp.is_connected():
            # Читаем команду из консоли
            try:
                cmd = input("> ").strip()
                if not cmd:
                    continue
                    
                # Парсим команду
                parts = cmd.split(' ')
                if len(parts) >= 2:
                    speeds = [int(x) for x in parts[0].strip('[]').split(',')]
                    servos = [int(x) for x in parts[1].strip('[]').split(',')]
                    
                    success, response = esp.sendMotionCommand(speeds, servos)
                    if success:
                        print(f"Response: Mode={response['mode']}, Values={response['values']}")
                    else:
                        print("No response from ESP")
                else:
                    print("Invalid format. Use: [a,b,c,d] [e,f,g,h]")
                    
            except ValueError:
                print("Invalid numbers")
            except EOFError:
                break
                
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        esp.close()