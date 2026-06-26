#!/usr/bin/env python3
"""
Главный модуль SLAM системы для исследования поля.

Логика работы:
1. Пока не исследованно все поле (3 зеленые и 3 синие/красные трубы):
   a. Проверяем, стоим ли на перекрестке (на старте - да)
   b. Если да: запускаем image -> analyse -> buildGraph
   c. Выбираем непосещенную точку, куда можно попасть быстрее всего
   d. Убираем флаг перекрестка
   e. Если не на перекрестке: едем к целевой точке
   f. Вычисляем скорости для 4 моторов (2 левых, 2 правых)
   g. Как только пришли - поднимаем флаг перекрестка
   h. Если на перекрестке и есть команды движения - выполняем их
   i. Trapezoidal acceleration для всех движений

Все действия логируются. Отправка на ESP - один раз за цикл.
"""

import cv2
import numpy as np
import logging
import time
import os
from datetime import datetime
from typing import Tuple, Optional

# Импорт модулей проекта
from сommunicate import ESPCommunication
from buildGraph import BuildGraph
import analyse
from tools import constrain


# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================

class Config:
    """Конфигурация системы"""
    # Параметры движения
    MAX_SPEED = 1000  # Максимальная скорость моторов
    MIN_SPEED = 50     # Минимальная скорость
    ACCEL_STEP = 100  # Шаг разгона/торможения
    
    # Ускорение (трапеция)
    ACCEL_TIME_MS = 500      # Время разгона в мс
    CRUISE_TIME_MS = 1000   # Время крейсерской скорости
    DECEL_TIME_MS = 500     # Время торможения
    
    # Перекрестки
    CROSSROAD_STOP_TIME_MS = 500  # Остановка на перекрестке для поворота
    CROSSROAD_DETECT_THRESHOLD = 50  # Порог определения перекрестка (пиксели)
    
    # Исследование
    REQUIRED_GREEN_PIPES = 3
    REQUIRED_COLORED_PIPES = 3
    
    # Сетка
    GRID_ROWS = 4
    GRID_COLS = 4


# =============================================================================
# УПРАВЛЕНИЕ ДВИЖЕНИЕМ С ТРАПЕЦИЕВИДНЫМ УСКОРЕНИЕМ
# =============================================================================

class MotionController:
    """
    Контроллер движения с трапециевидным профилем скорости.
    """
    
    def __init__(self, config: type):
        self.config = config
        self.current_speed = 0
        self.target_speed = 0
        self.is_moving = False
        self.phase = 'idle'  # idle, accel, cruise, decel
        self.phase_start_time = 0
        self.is_stopping_for_turn = False
        self.turn_stop_start = 0
    
    def start_motion(self, speed: int = None):
        """Начать движение."""
        if speed is None:
            speed = self.config.MAX_SPEED
        
        self.target_speed = constrain(speed, self.config.MIN_SPEED, self.config.MAX_SPEED)
        self.is_moving = True
        self.phase = 'accel'
        self.phase_start_time = time.time() * 1000
        self.is_stopping_for_turn = False
        logging.info(f"Начало движения: скорость={self.target_speed}")
    
    def stop(self):
        """Остановить движение."""
        self.is_moving = False
        self.target_speed = 0
        self.phase = 'idle'
        self.current_speed = 0
        logging.info("Остановка движения")
    
    def request_turn_stop(self):
        """Запросить остановку для поворота."""
        self.is_stopping_for_turn = True
        self.turn_stop_start = time.time() * 1000
        logging.info("Запрошена остановка для поворота")
    
    def update(self) -> Tuple[bool, int, int]:
        """
        Обновить состояние движения.
        
        Returns:
            Tuple[bool, int, int]: (продолжать_движение, скорость_левых, скорость_правых)
        """
        if not self.is_moving:
            return False, 0, 0
        
        current_time = time.time() * 1000
        
        # Проверяем остановку для поворота
        if self.is_stopping_for_turn:
            elapsed = current_time - self.turn_stop_start
            if elapsed < self.config.CROSSROAD_STOP_TIME_MS:
                self.current_speed = max(0, self.current_speed - self.config.ACCEL_STEP)
                return True, self.current_speed, self.current_speed
            else:
                self.is_stopping_for_turn = False
                self.current_speed = 0
                return True, 0, 0
        
        # Обновляем фазу движения
        phase_time = current_time - self.phase_start_time
        
        if self.phase == 'accel':
            if phase_time >= self.config.ACCEL_TIME_MS:
                self.phase = 'cruise'
                self.phase_start_time = current_time
            else:
                progress = phase_time / self.config.ACCEL_TIME_MS
                self.current_speed = int(
                    self.config.MIN_SPEED + 
                    (self.target_speed - self.config.MIN_SPEED) * progress
                )
        
        elif self.phase == 'cruise':
            if phase_time >= self.config.CRUISE_TIME_MS:
                self.phase = 'decel'
                self.phase_start_time = current_time
            self.current_speed = self.target_speed
        
        elif self.phase == 'decel':
            if phase_time >= self.config.DECEL_TIME_MS:
                self.phase = 'idle'
                self.current_speed = 0
                self.is_moving = False
            else:
                progress = 1.0 - phase_time / self.config.DECEL_TIME_MS
                self.current_speed = int(
                    self.config.MIN_SPEED + 
                    (self.target_speed - self.config.MIN_SPEED) * progress
                )
        
        return self.is_moving, self.current_speed, self.current_speed


# =============================================================================
# ОБНАРУЖЕНИЕ ПЕРЕКРЕСТКОВ
# =============================================================================

class CrossroadDetector:
    """Детектор перекрестков на основе анализа изображения."""
    
    def __init__(self, threshold: int = 50):
        self.threshold = threshold
    
    def is_on_crossroad(self, unwrapped_frame: np.ndarray) -> bool:
        """Определить, находится ли робот на перекрестке."""
        h, w = unwrapped_frame.shape[:2]
        
        center_x, center_y = w // 2, h // 2
        region_size = min(w, h) // 4
        
        x1 = max(0, center_x - region_size // 2)
        y1 = max(0, center_y - region_size // 2)
        x2 = min(w, center_x + region_size // 2)
        y2 = min(h, center_y + region_size // 2)
        
        center_region = unwrapped_frame[y1:y2, x1:x2]
        
        if len(center_region.shape) == 3:
            gray = cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = center_region
        
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = cv2.countNonZero(edges)
        
        return edge_pixels > self.threshold
    
    def is_approaching_crossroad(self, unwrapped_frame: np.ndarray) -> bool:
        """Определить, приближаемся ли к перекрестку."""
        h = unwrapped_frame.shape[0]
        forward_region = unwrapped_frame[:h // 4, :]
        
        if len(forward_region.shape) == 3:
            gray = cv2.cvtColor(forward_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = forward_region
        
        edges = cv2.Canny(gray, 50, 150)
        return cv2.countNonZero(edges) > self.threshold * 2


# =============================================================================
# ВИЗУАЛИЗАЦИЯ
# =============================================================================

class Visualizer:
    """Визуализатор состояния системы."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        if enabled:
            cv2.namedWindow('Unwrapped', cv2.WINDOW_NORMAL)
    
    def show(self, unwrapped: np.ndarray, status_text: str):
        """Показать визуализацию."""
        if not self.enabled:
            return
        
        status_img = unwrapped.copy()
        y_pos = 30
        for line in status_text.split('\n'):
            cv2.putText(status_img, line, (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_pos += 30
        
        cv2.imshow('Unwrapped', status_img)
        cv2.waitKey(1)
    
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
        
        # Инициализация компонентов
        self.esp = ESPCommunication(debug=True)
        self.graph = BuildGraph(
            grid_rows=self.config.GRID_ROWS, 
            grid_cols=self.config.GRID_COLS
        )
        self.motion = MotionController(self.config)
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
        
        # Инициализация видео
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
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
            level=logging.INFO,
            encoding='utf-8'
        )
        
        logging.info("=== SLAM Controller инициализирован ===")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Захватить кадр из источника."""
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def analyze_current_sector(self):
        """Анализировать текущий сектор (заглушка - требует unwrapped изображение)."""
        logging.info(f"Анализ сектора {self.current_position}")
        # TODO: Добавить реальный анализ с unwrapped изображением
        self.graph.build_edges()
    
    def select_next_target(self) -> Optional[Tuple[int, int]]:
        """Выбрать следующую целевую точку."""
        if self.graph.is_field_explored(
            self.config.REQUIRED_GREEN_PIPES,
            self.config.REQUIRED_COLORED_PIPES
        ):
            logging.info("=== ПОЛЕ ИССЛЕДОВАНО ===")
            return None
        
        target = self.graph.get_nearest_unvisited(self.current_position)
        
        if target:
            distance, commands, path = self.graph.find_path_to(
                self.current_position, target
            )
            self.pending_commands = list(commands)
            logging.info(f"Выбрана цель: {target}, путь: {commands}")
            self.is_on_crossroad = False
        
        return target
    
    def process_crossroad(self):
        """Обработка перекрестка."""
        if self.pending_commands:
            cmd = self.pending_commands.pop(0)
            
            if cmd in ('R', 'L', 'A'):
                self.motion.request_turn_stop()
                self._execute_turn(cmd)
                self._update_direction(cmd)
            elif cmd.startswith('F'):
                steps = int(cmd[1:]) if len(cmd) > 1 else 1
                logging.info(f"Движение вперед: {steps} сегментов")
                if steps > 1:
                    self.pending_commands.insert(0, f"F{steps-1}")
            
            self.is_on_crossroad = False
        else:
            self.select_next_target()
    
    def _execute_turn(self, turn_cmd: str):
        """Выполнить поворот."""
        if turn_cmd == 'R':
            self.motor_speeds = [
                self.config.MAX_SPEED, self.config.MAX_SPEED,
                -self.config.MAX_SPEED, -self.config.MAX_SPEED
            ]
        elif turn_cmd == 'L':
            self.motor_speeds = [
                -self.config.MAX_SPEED, -self.config.MAX_SPEED,
                self.config.MAX_SPEED, self.config.MAX_SPEED
            ]
        elif turn_cmd == 'A':
            self.motor_speeds = [
                self.config.MAX_SPEED, self.config.MAX_SPEED,
                -self.config.MAX_SPEED, -self.config.MAX_SPEED
            ]
        
        logging.info(f"Поворот {turn_cmd}: моторы={self.motor_speeds}")
    
    def _update_direction(self, turn_cmd: str):
        """Обновить направление после поворота."""
        dir_map = {'U': 0, 'R': 1, 'D': 2, 'L': 3}
        inv_map = {0: 'U', 1: 'R', 2: 'D', 3: 'L'}
        
        current_idx = dir_map.get(self.current_direction, 0)
        
        if turn_cmd == 'R':
            new_idx = (current_idx + 1) % 4
        elif turn_cmd == 'L':
            new_idx = (current_idx - 1) % 4
        elif turn_cmd == 'A':
            new_idx = (current_idx + 2) % 4
        
        self.current_direction = inv_map[new_idx]
        logging.info(f"Направление обновлено: {self.current_direction}")
    
    def send_command_and_wait(self) -> bool:
        """
        Отправить команду на ESP и дождаться ответа.
        
        Returns:
            bool: True если ответ получен
        """
        speeds = self.motor_speeds
        servos = [0, 0, 0, 0]
        
        success, response = self.esp.sendMotionCommand(speeds, servos)
        
        if not success:
            logging.warning(f"ESP не ответил на команду: speeds={speeds}")
        
        # Проверяем статус соединения
        if not self.esp.is_connected():
            logging.error("Соединение с ESP потеряно!")
            return False
        
        return success
    
    def main_loop(self):
        """Главный цикл системы."""
        logging.info("=== НАЧАЛО ГЛАВНОГО ЦИКЛА ===")
        
        cycle_count = 0
        frame = None
        
        try:
            while self.esp.is_connected():
                cycle_start = time.time()
                cycle_count += 1
                
                # Захватываем кадр
                frame = self.capture_frame()
                if frame is None:
                    logging.warning("Не удалось захватить кадр")
                    time.sleep(0.1)
                    continue
                
                # Основная логика
                if self.is_on_crossroad:
                    if self.pending_commands:
                        self.process_crossroad()
                    else:
                        self.select_next_target()
                else:
                    # В движении - обновляем motion controller
                    is_moving, left, right = self.motion.update()
                    self.motor_speeds = [left, left, right, right]
                
                # --- ОТПРАВКА КОМАНДЫ И ОЖИДАНИЕ ОТВЕТА ---
                if not self.send_command_and_wait():
                    break  # Соединение потеряно
                
                # --- ВИЗУАЛИЗАЦИЯ ---
                if frame is not None:
                    # Показываем оригинальный кадр если unwrap не реализован
                    status = self._get_status_string(cycle_count)
                    self.visualizer.show(frame, status)
                
                # Цикл ~30fps
                cycle_time = (time.time() - cycle_start) * 1000
                if cycle_time < 33:
                    time.sleep((33 - cycle_time) / 1000)
                
                # Проверяем условие завершения
                if self.graph.is_field_explored(
                    self.config.REQUIRED_GREEN_PIPES,
                    self.config.REQUIRED_COLORED_PIPES
                ):
                    logging.info("=== ИССЛЕДОВАНИЕ ЗАВЕРШЕНО ===")
                    break
                
                # Проверка выхода
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    logging.info("Выход по нажатию ESC")
                    break
                
        except KeyboardInterrupt:
            logging.info("Выход по KeyboardInterrupt")
        
        finally:
            self.shutdown()
    
    def _get_status_string(self, cycle_count: int) -> str:
        """Получить строку статуса."""
        status = self.graph.get_status()
        esp_stats = self.esp.get_stats()
        
        lines = [
            f"Цикл: {cycle_count}",
            f"Поз: {self.current_position}, напр: {self.current_direction}",
            f"Цель: {self.target_sector}",
            f"Команды: {''.join(self.pending_commands[:10])}",
            f"Скорость: {self.motor_speeds}",
            f"Перекресток: {self.is_on_crossroad}",
            f"Исследов: {status['visited_sectors']}/{status['total_sectors']} "
            f"({status['exploration_percent']:.0f}%)",
            f"Трубы: зел={status['green_pipes']}, син={status['blue_pipes']}, "
            f"красн={status['red_pipes']}",
            f"ESP: sent={esp_stats['sent']}, recv={esp_stats['received']}, "
            f"missed={esp_stats['missed']}",
            f"Исследован: {status['is_explored']}"
        ]
        
        return '\n'.join(lines)
    
    def shutdown(self):
        """Корректное завершение работы."""
        logging.info("=== ЗАВЕРШЕНИЕ РАБОТЫ ===")
        
        # Останавливаем моторы
        self.motor_speeds = [0, 0, 0, 0]
        self.esp.sendMotionCommand(self.motor_speeds, [0, 0, 0, 0])
        
        # Закрываем соединение с ESP
        self.esp.close()
        
        # Закрываем видео
        if self.cap:
            self.cap.release()
        
        # Закрываем визуализацию
        self.visualizer.close()
        
        # Финальный статус
        status = self.graph.get_status()
        
        print("\n" + "="*50)
        print("ИССЛЕДОВАНИЕ ЗАВЕРШЕНО")
        print("="*50)
        print(f"Исследованно секторов: {status['visited_sectors']}/{status['total_sectors']}")
        print(f"Найдено зеленых труб: {status['green_pipes']}")
        print(f"Найдено синих труб: {status['blue_pipes']}")
        print(f"Найдено красных труб: {status['red_pipes']}")
        print("="*50)


# =============================================================================
# ТОЧКА ВХОДА
# =============================================================================

def main():
    """Главная функция."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SLAM Controller')
    parser.add_argument('--video', '-v', type=str, default=None,
                       help='Путь к видеофайлу для отладки')
    
    args = parser.parse_args()
    
    controller = SLAMController(video_path=args.video)
    controller.main_loop()


if __name__ == "__main__":
    main()