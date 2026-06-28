#!/usr/bin/env python3
"""
Главный модуль SLAM системы для исследования поля.

Логика работы:
1. Пока не исследованно все поле (3 зеленые и 3 синие/красные трубы):
   a. Проверяем, стоим ли на перекрестке (на старте - да)
   b. Если да: запускаем analyse -> graph_builder
   c. Выбираем непосещенную точку, куда можно попасть быстрее всего
   d. Убираем флаг перекрестка
   e. Если не на перекрестке: едем к целевой точке
   f. Вычисляем скорости для 4 моторов (2 левых, 2 правых)
   g. Как только пришли - поднимаем флаг перекрестка
   h. Если на перекрестке и есть команды движения - выполняем их
   i. Trapezoidal acceleration для всех движений

Все действия логируются. Отправка на ESP - один раз за цикл.
"""

# =============================================================================
# DEBUG FLAG - Set to True for verbose debug output and image saving
# =============================================================================
DEBUG = True
DEBUG_SAVE_IMAGES = True  # Save debug images to Output/debug/
DEBUG_SAVE_INTERVAL = 10   # Save every N frames


import cv2
import numpy as np
import logging
import time
import os
from datetime import datetime
from typing import Tuple, Optional

# Импорт модулей проекта
from esp_comms import ESPCommunication
from graph_builder import BuildGraph
from sector_analyzer import analyze_image
from robot_control import RobotController, RobotConfig, Direction, TurnType
from utils import constrain


# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================

class Config:
    """Конфигурация системы. Использует RobotConfig для параметров движения."""
    # DEBUG режим
    DEBUG_MODE = DEBUG
    
    # Исследование
    REQUIRED_GREEN_PIPES = 3
    REQUIRED_COLORED_PIPES = 3
    
    # Сетка
    GRID_ROWS = 4
    GRID_COLS = 4
    
    # Перекрестки - детекция
    CROSSROAD_DETECT_THRESHOLD = 50  # Порог определения перекрестка (пиксели)
    
    # Параметры движения - делегируем в RobotConfig
    MAX_SPEED = RobotConfig.MAX_MOTOR_SPEED
    MIN_SPEED = RobotConfig.MIN_MOTOR_SPEED
    ACCEL_TIME_MS = RobotConfig.ACCEL_TIME_MS
    CRUISE_TIME_MS = RobotConfig.CRUISE_TIME_MS
    DECEL_TIME_MS = RobotConfig.DECEL_TIME_MS
    CROSSROAD_STOP_TIME_MS = RobotConfig.CROSSROAD_STOP_TIME_MS
    
    # Тики энкодера
    TICK_PER_90_TURN = RobotConfig.TICK_PER_90_TURN


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ DEBUG
# =============================================================================

def ensure_debug_dirs():
    """Создать директории для debug файлов."""
    os.makedirs('Output/debug', exist_ok=True)
    os.makedirs('Output/frames', exist_ok=True)
    os.makedirs('Output/analyzed', exist_ok=True)
    os.makedirs('Output/status', exist_ok=True)
    os.makedirs('Output/crossroads', exist_ok=True)


def save_debug_image(image, name, subfolder='debug'):
    """Сохранить debug изображение."""
    if not DEBUG_SAVE_IMAGES:
        return
    
    path = f"Output/{subfolder}/{name}_{datetime.now().strftime('%H%M%S_%f')}.jpg"
    cv2.imwrite(path, image)
    logging.debug(f"DEBUG: Saved image: {path}")


def log_debug(text):
    """Вывести debug текст если DEBUG=True."""
    if DEBUG:
        print(f"[DEBUG] {text}")
        logging.debug(text)


def log_verbose(label, data):
    """Вывести подробные данные если DEBUG=True."""
    if DEBUG:
        if isinstance(data, dict):
            items = ', '.join([f"{k}={v}" for k, v in data.items()])
            print(f"[VERBOSE] {label}: {items}")
        else:
            print(f"[VERBOSE] {label}: {data}")
        logging.debug(f"{label}: {data}")




# =============================================================================
# ОБНАРУЖЕНИЕ ПЕРЕКРЕСТКОВ
# =============================================================================

class CrossroadDetector:
    """Детектор перекрестков на основе анализа изображения."""
    
    def __init__(self, threshold: int = 50):
        self.threshold = threshold
        log_debug(f"CrossroadDetector: initialized with threshold={threshold}")
    
    def is_on_crossroad(self, frame: np.ndarray) -> Tuple[bool, dict]:
        """
        Определить, находится ли робот на перекрестке.
        
        Returns:
            Tuple[bool, dict]: (результат, debug_data)
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        h, w = gray.shape
        
        center_x, center_y = w // 2, h // 2
        region_size = min(w, h) // 4
        
        x1 = max(0, center_x - region_size // 2)
        y1 = max(0, center_y - region_size // 2)
        x2 = min(w, center_x + region_size // 2)
        y2 = min(h, center_y + region_size // 2)
        
        center_region = gray[y1:y2, x1:x2]
        
        edges = cv2.Canny(center_region, 50, 150)
        edge_pixels = cv2.countNonZero(edges)
        
        result = edge_pixels > self.threshold
        
        debug_data = {
            'edge_pixels': edge_pixels,
            'threshold': self.threshold,
            'region_size': region_size,
            'center': (center_x, center_y)
        }
        
        log_verbose("CrossroadDetector.is_on_crossroad", debug_data)
        
        # Сохраняем debug изображение
        if DEBUG_SAVE_IMAGES:
            debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(debug_img, f"Crossroad: {result} (edges={edge_pixels})", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            save_debug_image(debug_img, "crossroad_detect", "crossroads")
        
        return result, debug_data
    
    def is_approaching_crossroad(self, frame: np.ndarray) -> bool:
        """Определить, приближаемся ли к перекрестку."""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        h = gray.shape[0]
        forward_region = gray[:h // 4, :]
        
        edges = cv2.Canny(forward_region, 50, 150)
        result = cv2.countNonZero(edges) > self.threshold * 2
        
        log_debug(f"CrossroadDetector: approaching={result}")
        return result


# =============================================================================
# ВИЗУАЛИЗАЦИЯ И DEBUG
# =============================================================================

class Visualizer:
    """Визуализатор состояния системы с DEBUG поддержкой."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and not DEBUG  # В DEBUG режиме не показываем окна
        self.frame_count = 0
        
        if self.enabled:
            cv2.namedWindow('Unwrapped', cv2.WINDOW_NORMAL)
        
        log_debug(f"Visualizer: enabled={enabled}, actual_enabled={self.enabled}")
    
    def show(self, frame: np.ndarray, status_text: str, extra_debug: dict = None):
        """Показать визуализацию."""
        if self.enabled:
            status_img = frame.copy()
            y_pos = 30
            for line in status_text.split('\n'):
                cv2.putText(status_img, line, (10, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_pos += 30
            cv2.imshow('Unwrapped', status_img)
            cv2.waitKey(1)
        
        # Всегда сохраняем debug изображения
        if DEBUG_SAVE_IMAGES:
            self.frame_count += 1
            if self.frame_count % DEBUG_SAVE_INTERVAL == 0:
                debug_img = frame.copy()
                if len(debug_img.shape) == 2:
                    debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)
                
                y_pos = 30
                for line in status_text.split('\n'):
                    cv2.putText(debug_img, line, (10, y_pos), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y_pos += 20
                
                # Добавляем extra debug info
                if extra_debug:
                    y_pos += 20
                    for key, value in extra_debug.items():
                        cv2.putText(debug_img, f"{key}: {value}", (10, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                        y_pos += 15
                
                save_debug_image(debug_img, f"frame_{self.frame_count}", "frames")
    
    def save_status(self, status_text: str, cycle: int):
        """Сохранить статус в текстовый файл."""
        if not DEBUG_SAVE_IMAGES:
            return
        
        path = f"Output/status/status_{cycle:06d}.txt"
        with open(path, 'w') as f:
            f.write(f"=== SLAM Status (Cycle {cycle}) ===\n")
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
            f.write("=" * 40 + "\n")
            f.write(status_text)
        log_debug(f"Status saved: {path}")
    
    def close(self):
        """Закрыть все окна."""
        if self.enabled:
            cv2.destroyAllWindows()


# =============================================================================
# ГЛАВНЫЙ КЛАСС СИСТЕМЫ
# =============================================================================

class SLAMController:
    """
    Главный контроллер SLAM системы.
    """
    
    def __init__(self, video_path: str = None):
        """
        Инициализация системы.
        
        Args:
            video_path: Путь к видеофайлу (для отладки)
        """
        self.config = Config
        self.video_path = video_path
        
        # Создаем debug директории
        if DEBUG:
            ensure_debug_dirs()
            print("=" * 60)
            print("DEBUG MODE ENABLED")
            print("Images will be saved to Output/debug/, Output/frames/")
            print("=" * 60)
        
        # Инициализация компонентов
        self.esp = ESPCommunication(debug=DEBUG)
        self.graph = BuildGraph(
            grid_rows=self.config.GRID_ROWS, 
            grid_cols=self.config.GRID_COLS
        )
        
        # RobotController для управления роботом (новый модуль)
        self.robot = RobotController(
            self.esp,
            config=RobotConfig,
            debug=DEBUG
        )
        
        self.crossroad_detector = CrossroadDetector(
            threshold=self.config.CROSSROAD_DETECT_THRESHOLD
        )
        self.visualizer = Visualizer(enabled=True)
        
        # Состояние системы
        self.is_on_crossroad = True  # На старте - на перекрестке
        self.current_position = (0, 0)  # (row, col)
        self.current_direction = 'U'  # U, D, L, R
        self.target_sector = None
        self.pending_commands = []
        self.motor_speeds = [0, 0, 0, 0]
        self.servo_positions = [RobotConfig.SERVO_STRAIGHT] * 4
        self.cycle_count = 0
        
        # Инициализация видео
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            log_debug(f"Video capture initialized: {video_path}")
        else:
            self.cap = None
        
        # Логирование
        timestamp = datetime.now().strftime('%d_%H-%M-%S')
        os.makedirs('Output/Logs', exist_ok=True)
        log_file = f'Output/Logs/app_{timestamp}.log'
        
        logging.basicConfig(
            filename=log_file,
            filemode="w",
            format='%(asctime)s | %(levelname)-8s | %(message)s',
            level=logging.DEBUG if DEBUG else logging.INFO,
            encoding='utf-8'
        )
        
        logging.info("=" * 60)
        logging.info("=== SLAM Controller INITIALIZED ===")
        logging.info(f"DEBUG mode: {DEBUG}")
        logging.info(f"Video path: {video_path}")
        logging.info(f"Grid size: {self.config.GRID_ROWS}x{self.config.GRID_COLS}")
        logging.info(f"Required pipes: green={self.config.REQUIRED_GREEN_PIPES}, "
                    f"colored={self.config.REQUIRED_COLORED_PIPES}")
        logging.info("=" * 60)
        
        if DEBUG:
            print(f"[INIT] SLAM Controller initialized")
            print(f"[INIT] Log file: {log_file}")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Захватить кадр из источника."""
        if self.cap is None:
            log_debug("No video source, returning None")
            return None
        
        ret, frame = self.cap.read()
        if ret:
            log_debug(f"Frame captured: {frame.shape}")
        else:
            logging.warning("Failed to capture frame")
            log_debug("Frame capture failed")
        return frame if ret else None
    
    def analyze_current_sector(self, frame: np.ndarray):
        """Анализировать текущий сектор."""
        logging.info(f"Analyzing sector {self.current_position}")
        log_debug(f"Analyze: starting analysis for sector {self.current_position}")
        
        # Анализируем изображение
        analysis = analyze_image(self.video_path) if self.video_path else None
        
        if analysis:
            # Добавляем в граф
            self.graph.add_sector_analysis(
                self.current_position[0],
                self.current_position[1],
                analysis
            )
            
            logging.info(f"Analysis complete: {analysis}")
            log_verbose("Analysis results", analysis)
            
            # Сохраняем анализ
            if DEBUG_SAVE_IMAGES and frame is not None:
                analysis_img = frame.copy()
                h, w = analysis_img.shape[:2]
                
                # Рисуем информацию об анализе
                y_pos = 30
                for key, value in analysis.items():
                    if key != 'mask_green':
                        text = f"{key}: {value}"
                        cv2.putText(analysis_img, text, (10, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        y_pos += 20
                
                save_debug_image(analysis_img, f"analysis_{self.current_position[0]}_{self.current_position[1]}", "analyzed")
        
        # Строим ребра графа
        self.graph.build_edges()
        log_debug("Graph edges rebuilt")
    
    def select_next_target(self) -> Optional[Tuple[int, int]]:
        """Выбрать следующую целевую точку."""
        if self.graph.is_field_explored(
            self.config.REQUIRED_GREEN_PIPES,
            self.config.REQUIRED_COLORED_PIPES
        ):
            logging.info("=" * 40)
            logging.info("=== ПОЛЕ ПОЛНОСТЬЮ ИССЛЕДОВАНО ===")
            logging.info("=" * 40)
            return None
        
        target = self.graph.get_nearest_unvisited(self.current_position)
        
        if target:
            distance, commands, path = self.graph.find_path_to(
                self.current_position, target
            )
            self.pending_commands = list(commands)
            
            logging.info(f"Target selected: {target}")
            logging.info(f"Path: {commands} (distance={distance}, path={path})")
            log_verbose("Route planning", {
                'from': self.current_position,
                'to': target,
                'distance': distance,
                'commands': commands,
                'path': path
            })
            
            self.is_on_crossroad = False
        
        return target
    
    def process_crossroad(self):
        """Обработка перекрестка - передаем команды RobotController."""
        if self.pending_commands:
            # Преобразуем команды в формат RobotController
            commands = self._convert_commands_for_robot(self.pending_commands)
            self.robot.set_commands(commands)
            self.pending_commands.clear()
            
            logging.info(f"Crossroad: commands sent to RobotController")
            log_debug(f"RobotController commands: {commands}")
            
            self.is_on_crossroad = False
        else:
            self.select_next_target()
    
    def _convert_commands_for_robot(self, commands: list) -> list:
        """
        Преобразовать команды в формат RobotController.
        
        Args:
            commands: Список команд ['F', 'F2', 'R', 'L', 'A']
            
        Returns:
            Список команд для RobotController
        """
        result = []
        for cmd in commands:
            cmd = cmd.strip().upper()
            if not cmd:
                continue
            result.append(cmd)
        return result
    
    def send_command_and_wait(self) -> bool:
        """
        Обновить состояние робота через RobotController.
        
        Returns:
            bool: True если робот в рабочем состоянии
        """
        # Обновляем RobotController - он сам отправляет команды на ESP
        success = self.robot.update()
        
        # Синхронизируем состояние с нашим контроллером
        self.motor_speeds = self.robot.motor_speeds
        self.servo_positions = self.robot.servo_positions
        self.current_direction = self.robot.direction.value
        
        log_debug(f"ESP TX: speeds={self.motor_speeds}, servos={self.servo_positions}")
        
        # Проверяем, завершены ли все команды
        if not self.robot.has_commands() and self.robot.state.name == 'IDLE':
            if not self.is_on_crossroad:
                self.is_on_crossroad = True
                logging.info("Crossroad reached!")
                # Анализируем текущий сектор
                frame = self.capture_frame()
                if frame is not None:
                    self.analyze_current_sector(frame)
        
        return success
    
    
    def main_loop(self):
        """Главный цикл системы."""
        logging.info("=" * 60)
        logging.info("=== MAIN LOOP STARTED ===")
        logging.info("=" * 60)
        
        if DEBUG:
            print("[MAIN] Starting main loop...")
        
        frame = None
        
        try:
            while self.esp.is_connected():
                cycle_start = time.time()
                self.cycle_count += 1
                
                # Захватываем кадр
                frame = self.capture_frame()
                if frame is None:
                    logging.warning("No frame captured, retrying...")
                    time.sleep(0.1)
                    continue
                
                # Debug: сохраняем каждый кадр периодически
                if DEBUG_SAVE_IMAGES and self.cycle_count % DEBUG_SAVE_INTERVAL == 0:
                    save_debug_image(frame, f"raw_frame_{self.cycle_count}", "frames")
                
                # Основная логика
                if self.is_on_crossroad:
                    log_debug("State: ON_CROSSROAD")
                    if self.pending_commands:
                        self.process_crossroad()
                    else:
                        self.select_next_target()
                else:
                    log_debug("State: MOVING")
                    # RobotController управляет движением автоматически
                    pass
                
                # Определяем перекресток
                is_crossroad, crossroad_debug = self.crossroad_detector.is_on_crossroad(frame)
                
                # --- ОТПРАВКА КОМАНДЫ И ОЖИДАНИЕ ОТВЕТА ---
                if not self.send_command_and_wait():
                    break  # Соединение потеряно
                
                # --- СТАТУС И ВИЗУАЛИЗАЦИЯ ---
                status = self._get_status_string()
                esp_stats = self.esp.get_stats()
                
                extra_debug = {
                    'crossroad': is_crossroad,
                    'edge_pixels': crossroad_debug.get('edge_pixels', 0),
                    'esp_connected': self.esp.is_connected(),
                    'missed': esp_stats.get('missed', 0)
                }
                
                self.visualizer.show(frame, status, extra_debug)
                self.visualizer.save_status(status, self.cycle_count)
                
                # Цикл ~30fps
                cycle_time = (time.time() - cycle_start) * 1000
                logging.debug(f"Cycle {self.cycle_count}: {cycle_time:.1f}ms")
                
                if cycle_time < 33:
                    time.sleep((33 - cycle_time) / 1000)
                
                # Проверяем условие завершения
                if self.graph.is_field_explored(
                    self.config.REQUIRED_GREEN_PIPES,
                    self.config.REQUIRED_COLORED_PIPES
                ):
                    logging.info("=" * 40)
                    logging.info("=== EXPLORATION COMPLETE ===")
                    logging.info("=" * 40)
                    break
                
                # Проверка выхода (только если не DEBUG, иначе ждем)
                if not DEBUG:
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        logging.info("Exit: ESC pressed")
                        break
                
        except KeyboardInterrupt:
            logging.info("Exit: KeyboardInterrupt")
        
        finally:
            self.shutdown()
    
    def _get_status_string(self) -> str:
        """Получить строку статуса."""
        status = self.graph.get_status()
        esp_stats = self.esp.get_stats()
        robot_status = self.robot.get_status()
        
        lines = [
            f"=== SLAM Controller ===",
            f"Cycle: {self.cycle_count}",
            f"Position: {self.current_position}, Dir: {self.current_direction}",
            f"Target: {self.target_sector}",
            f"On Crossroad: {self.is_on_crossroad}",
            f"--- Robot Controller ---",
            f"State: {robot_status['state']}",
            f"Motors: {self.motor_speeds}",
            f"Servos: {self.servo_positions}",
            f"Commands: {self.robot.get_commands_string()[:15]}...",
            f"Queue: {robot_status['commands_in_queue']}",
            f"--- Exploration ---",
            f"Explored: {status['visited_sectors']}/{status['total_sectors']} ({status['exploration_percent']:.0f}%)",
            f"Pipes: G={status['green_pipes']}, B={status['blue_pipes']}, R={status['red_pipes']}",
            f"--- ESP ---",
            f"sent={esp_stats['sent']}, recv={esp_stats['received']}, missed={esp_stats['missed']}",
            f"--- Status ---",
            f"Is Explored: {status['is_explored']}",
            f"DEBUG: {DEBUG}",
        ]
        
        return '\n'.join(lines)
    
    def shutdown(self):
        """Корректное завершение работы."""
        logging.info("=" * 60)
        logging.info("=== SHUTDOWN ===")
        logging.info("=" * 60)
        
        # Останавливаем робота через RobotController
        self.robot.reset()
        self.robot.stop_motors()
        self.robot.set_servos_straight()
        self.robot.send_command()
        
        # Закрываем соединение с ESP
        self.esp.close()
        
        # Закрываем видео
        if self.cap:
            self.cap.release()
        
        # Закрываем визуализацию
        self.visualizer.close()
        
        # Финальный статус
        status = self.graph.get_status()
        esp_stats = self.esp.get_stats()
        robot_status = self.robot.get_status()
        
        summary = f"""
================================================================
                    SLAM EXPLORATION COMPLETE
================================================================
Sectors explored: {status['visited_sectors']}/{status['total_sectors']} ({status['exploration_percent']:.1f}%)
Green pipes found: {status['green_pipes']}
Blue pipes found: {status['blue_pipes']}
Red pipes found: {status['red_pipes']}
Ramps found: {status['ramps']}

Robot Control:
  State: {robot_status['state']}
  Direction: {robot_status['direction']}
  Position: {robot_status['position']}
  Commands executed: {robot_status['command_count']}
  Errors: {robot_status['error_count']}

ESP Communication:
  Commands sent: {esp_stats['sent']}
  Responses received: {esp_stats['received']}
  Missed responses: {esp_stats['missed']}
  Success rate: {esp_stats['success_rate']:.1f}%

Total cycles: {self.cycle_count}
DEBUG mode: {DEBUG}
Output saved to: Output/debug/, Output/frames/, Output/status/
================================================================
"""
        print(summary)
        logging.info(summary)


# =============================================================================
# ТОЧКА ВХОДА
# =============================================================================

def main():
    """Главная функция."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SLAM Controller')
    parser.add_argument('--video', '-v', type=str, default=None,
                       help='Path to video file for testing')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SLAM Controller")
    print("=" * 60)
    print(f"DEBUG mode: {DEBUG}")
    print(f"Save images: {DEBUG_SAVE_IMAGES}")
    print(f"Video file: {args.video}")
    print("=" * 60)
    
    controller = SLAMController(video_path=args.video)
    controller.main_loop()


if __name__ == "__main__":
    main()