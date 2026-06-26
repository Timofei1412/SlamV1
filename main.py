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
from typing import Tuple, Optional, List

# Импорт модулей проекта
from сommunicate import ESPCommunication
from localisation import ConicalLocalization
from buildGraph import BuildGraph, FloorLevel
from router import Pathfinder
import image as image_module
import analyse
from testing import CrossTracker, detect_raw_crosses, cluster_and_average, build_grid_from_points
from plane import build_combined_maps, remap_frame
from tools import constrain


# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================

class Config:
    """Конфигурация системы"""
    # Параметры камеры и unwrap
    CAMERA_CX = 308
    CAMERA_CY = 234
    CAMERA_OUTER_R = 230
    CAMERA_LENS_DEG = -81.86
    CAMERA_CONE_POWER = 2.245
    CAMERA_ROTATION_DEG = -2.0
    CAMERA_TOP_SIZE = 640
    CAMERA_FIELD_SCALE = 0.70
    
    # ROI
    ROI_PATH = 'Images/1.png'
    
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
    
    # Гусеницы
    LEFT_MOTORS = [0, 1]   # Индексы левых моторов
    RIGHT_MOTORS = [2, 3]  # Индексы правых моторов
    
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
    
    Профиль движения (трапеция):
    1. Разгон (ACCEL_TIME_MS)
    2. Крейсерская скорость (CRUISE_TIME_MS)
    3. Торможение (DECEL_TIME_MS)
    
    При обнаружении перекрестка - торможение.
    При повороте - остановка на CROSSROAD_STOP_TIME_MS.
    """
    
    def __init__(self, config: type):
        self.config = config
        self.current_speed = 0
        self.target_speed = 0
        self.is_moving = False
        self.phase = 'idle'  # idle, accel, cruise, decel, stop
        self.phase_start_time = 0
        self.is_stopping_for_turn = False
        self.turn_stop_start = 0
    
    def start_motion(self, direction: float, speed: int = None):
        """
        Начать движение в направлении.
        
        Args:
            direction: Угол направления в градусах (0 = вправо, 90 = вверх)
            speed: Целевая скорость (если None - максимальная)
        """
        if speed is None:
            speed = self.config.MAX_SPEED
        
        self.target_speed = constrain(speed, self.config.MIN_SPEED, self.config.MAX_SPEED)
        self.is_moving = True
        self.phase = 'accel'
        self.phase_start_time = time.time() * 1000
        self.is_stopping_for_turn = False
        logging.info(f"Начало движения: напр={direction}°, скорость={self.target_speed}")
    
    def stop(self):
        """Остановить движение."""
        self.is_moving = False
        self.target_speed = 0
        self.phase = 'idle'
        self.current_speed = 0
        logging.info("Остановка движения")
    
    def request_turn_stop(self):
        """Запросить остановку для поворота на перекрестке."""
        self.is_stopping_for_turn = True
        self.turn_stop_start = time.time() * 1000
        logging.info("Запрошена остановка для поворота")
    
    def update(self, delta_time_ms: float) -> Tuple[bool, int, int]:
        """
        Обновить состояние движения.
        
        Args:
            delta_time_ms: Время с последнего обновления в мс
            
        Returns:
            Tuple[bool, int, int]: (продолжать_движение, скорость_левых, скорость_правых)
        """
        if not self.is_moving:
            return False, 0, 0
        
        current_time = time.time() * 1000
        
        # Проверяем, нужно ли ждать остановку для поворота
        if self.is_stopping_for_turn:
            elapsed = current_time - self.turn_stop_start
            if elapsed < self.config.CROSSROAD_STOP_TIME_MS:
                # Тормозим
                self.current_speed = max(0, self.current_speed - self.config.ACCEL_STEP)
                return True, self._calc_speeds(self.current_speed)
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
                phase_time = 0
            else:
                progress = phase_time / self.config.ACCEL_TIME_MS
                self.current_speed = int(self.config.MIN_SPEED + 
                                        (self.target_speed - self.config.MIN_SPEED) * progress)
        
        elif self.phase == 'cruise':
            if phase_time >= self.config.CRUISE_TIME_MS:
                self.phase = 'decel'
                self.phase_start_time = current_time
                phase_time = 0
            self.current_speed = self.target_speed
        
        elif self.phase == 'decel':
            if phase_time >= self.config.DECEL_TIME_MS:
                self.phase = 'idle'
                self.current_speed = 0
                self.is_moving = False
            else:
                progress = 1.0 - phase_time / self.config.DECEL_TIME_MS
                self.current_speed = int(self.config.MIN_SPEED + 
                                        (self.target_speed - self.config.MIN_SPEED) * progress)
        
        return self.is_moving, self._calc_speeds(self.current_speed)
    
    def _calc_speeds(self, base_speed: int) -> Tuple[int, int]:
        """
        Рассчитать скорости для левых и правых моторов.
        
        Args:
            base_speed: Базовая скорость
            
        Returns:
            Tuple[int, int]: (скорость_левых, скорость_правых)
        """
        # Пока просто возвращаем одинаковые скорости
        # Повороты обрабатываются отдельно через команды ESP
        return base_speed, base_speed
    
    def should_stop_at_crossroad(self, approaching_crossroad: bool) -> bool:
        """
        Определить, нужно ли останавливаться на перекрестке.
        
        Args:
            approaching_crossroad: Приближаемся ли к перекрестку
            
        Returns:
            bool: True если нужно остановиться
        """
        if not approaching_crossroad:
            return False
        
        # Если мы в фазе торможения и приближаемся к перекрестку
        if self.phase in ('cruise', 'accel'):
            return True
        
        return False
    
    def get_decel_distance(self) -> float:
        """
        Получить расстояние торможения.
        
        Returns:
            float: Расстояние в условных единицах
        """
        # Упрощенный расчет
        return (self.current_speed ** 2) / (2 * self.config.ACCEL_STEP) if self.current_speed > 0 else 0


# =============================================================================
# ОБНАРУЖЕНИЕ ПЕРЕКРЕСТКОВ
# =============================================================================

class CrossroadDetector:
    """
    Детектор перекрестков на основе анализа изображения.
    
    Перекресток определяется по наличию пересекающихся линий
    в центральной части unwrapped изображения.
    """
    
    def __init__(self, threshold: int = 50):
        self.threshold = threshold
        self.prev_crossroad_state = False
    
    def is_on_crossroad(self, unwrapped_frame: np.ndarray) -> bool:
        """
        Определить, находится ли робот на перекрестке.
        
        Args:
            unwrapped_frame: Unwrapped изображение
            
        Returns:
            bool: True если на перекрестке
        """
        h, w = unwrapped_frame.shape[:2]
        
        # Берем центральную область
        center_x, center_y = w // 2, h // 2
        region_size = min(w, h) // 4
        
        x1 = max(0, center_x - region_size // 2)
        y1 = max(0, center_y - region_size // 2)
        x2 = min(w, center_x + region_size // 2)
        y2 = min(h, center_y + region_size // 2)
        
        center_region = unwrapped_frame[y1:y2, x1:x2]
        
        # Конвертируем в градации серого
        if len(center_region.shape) == 3:
            gray = cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = center_region
        
        # Находим края
        edges = cv2.Canny(gray, 50, 150)
        
        # Считаем количество белых пикселей (краев)
        edge_pixels = cv2.countNonZero(edges)
        
        is_crossroad = edge_pixels > self.threshold
        self.prev_crossroad_state = is_crossroad
        
        return is_crossroad
    
    def is_approaching_crossroad(self, unwrapped_frame: np.ndarray) -> bool:
        """
        Определить, приближаемся ли к перекрестку.
        
        Args:
            unwrapped_frame: Unwrapped изображение
            
        Returns:
            bool: True если приближаемся
        """
        h, w = unwrapped_frame.shape[:2]
        
        # Анализируем область впереди робота
        forward_region_h = h // 4
        
        # Верхняя часть изображения - это "вперед"
        forward_region = unwrapped_frame[:forward_region_h, :]
        
        if len(forward_region.shape) == 3:
            gray = cv2.cvtColor(forward_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = forward_region
        
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = cv2.countNonZero(edges)
        
        return edge_pixels > self.threshold * 2


# =============================================================================
# ВИЗУАЛИЗАЦИЯ
# =============================================================================

class Visualizer:
    """Визуализатор состояния системы."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        if enabled:
            cv2.namedWindow('SLAM Control', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Unwrapped', cv2.WINDOW_NORMAL)
    
    def show(self, unwrapped: np.ndarray, status_text: str):
        """
        Показать визуализацию.
        
        Args:
            unwrapped: Unwrapped изображение
            status_text: Текст статуса
        """
        if not self.enabled:
            return
        
        # Добавляем текст статуса
        h, w = unwrapped.shape[:2]
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
    
    Управляет исследованием поля:
    - Перемещение между перекрестками
    - Анализ секторов
    - Построение графа
    - Планирование маршрута
    """
    
    def __init__(self, video_path: str = None, use_camera: bool = False):
        """
        Инициализация системы.
        
        Args:
            video_path: Путь к видеофайлу (для отладки)
            use_camera: Использовать реальную камеру
        """
        self.config = Config
        self.use_camera = use_camera
        self.video_path = video_path
        
        # Инициализация компонентов
        self.esp = ESPCommunication(debug=False)
        self.localization = None  # Инициализируется при первом кадре
        self.graph = BuildGraph(grid_rows=self.config.GRID_ROWS, 
                               grid_cols=self.config.GRID_COLS)
        self.tracker = CrossTracker()
        self.motion = MotionController(self.config)
        self.crossroad_detector = CrossroadDetector(
            threshold=self.config.CROSSROAD_DETECT_THRESHOLD
        )
        self.visualizer = Visualizer(enabled=True)
        
        # Состояние системы
        self.is_on_crossroad = True  # На старте - на перекрестке
        self.current_position = (0, 0)  # (row, col)
        self.current_direction = 'U'  # U, D, L, R
        self.target_sector = None  # Целевой сектор
        self.pending_commands = []  # Команды для выполнения
        self.motor_speeds = [0, 0, 0, 0]  # Текущие скорости моторов
        
        # Remap карты
        self.map_x = None
        self.map_y = None
        self.roi_mask = None
        
        # Инициализация видео
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
        elif use_camera:
            self.cap = cv2.VideoCapture(0)
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
        logging.info(f"Требуется труб: зеленых={self.config.REQUIRED_GREEN_PIPES}, "
                    f"цветных={self.config.REQUIRED_COLORED_PIPES}")
    
    def initialize_remap(self, frame: np.ndarray):
        """
        Инициализировать remap карты для unwrapping.
        
        Args:
            frame: Первый кадр для определения размера
        """
        h, w = frame.shape[:2]
        
        self.map_x, self.map_y = build_combined_maps(
            top_size=self.config.CAMERA_TOP_SIZE,
            source_width=w,
            source_height=h,
            cx=self.config.CAMERA_CX,
            cy=self.config.CAMERA_CY,
            outer_r=self.config.CAMERA_OUTER_R,
            rotation_deg=self.config.CAMERA_ROTATION_DEG,
            field_scale=self.config.CAMERA_FIELD_SCALE,
            lens_deg=self.config.CAMERA_LENS_DEG,
            cone_power=self.config.CAMERA_CONE_POWER
        )
        
        # Загружаем ROI маску
        if os.path.exists(self.config.ROI_PATH):
            roi_raw = cv2.imread(self.config.ROI_PATH, cv2.IMREAD_GRAYSCALE)
            if roi_raw is not None:
                self.roi_mask = cv2.resize(roi_raw, 
                    (self.config.CAMERA_TOP_SIZE, self.config.CAMERA_TOP_SIZE),
                    interpolation=cv2.INTER_NEAREST)
                _, self.roi_mask = cv2.threshold(self.roi_mask, 127, 255, cv2.THRESH_BINARY)
                logging.info(f"ROI маска загружена: {self.config.ROI_PATH}")
        
        # Инициализируем localization
        self.localization = ConicalLocalization(
            cx=self.config.CAMERA_CX,
            cy=self.config.CAMERA_CY,
            outer_r=self.config.CAMERA_OUTER_R,
            lens_deg=self.config.CAMERA_LENS_DEG,
            cone_power=self.config.CAMERA_CONE_POWER,
            rotation_deg=self.config.CAMERA_ROTATION_DEG,
            top_size=self.config.CAMERA_TOP_SIZE,
            field_scale=self.config.CAMERA_FIELD_SCALE,
            roi=self.config.ROI_PATH,
            debug_mode=False
        )
        
        logging.info(f"Remap инициализирован: {w}x{h} -> {self.config.CAMERA_TOP_SIZE}x{self.config.CAMERA_TOP_SIZE}")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Захватить кадр из источника.
        
        Returns:
            np.ndarray или None
        """
        if self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def unwrap_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Развернуть кадр в top-down вид.
        
        Args:
            frame: Исходный кадр
            
        Returns:
            np.ndarray: Unwrapped изображение
        """
        if self.map_x is None:
            self.initialize_remap(frame)
        
        return remap_frame(frame, self.map_x, self.map_y, (0, 0, 0))
    
    def analyze_current_sector(self, unwrapped: np.ndarray):
        """
        Анализировать текущий сектор.
        
        Args:
            unwrapped: Unwrapped изображение
        """
        # TODO: Разбить unwrapped на секторы и анализировать текущий
        # Пока используем весь образ как один сектор
        
        # Сохраняем временное изображение для анализа
        os.makedirs('Output', exist_ok=True)
        temp_path = 'Output/current_sector.jpg'
        cv2.imwrite(temp_path, unwrapped)
        
        # Анализируем
        analysis = analyse.analyze_image(temp_path)
        
        if analysis:
            # Добавляем в граф
            self.graph.add_sector_analysis(
                self.current_position[0],
                self.current_position[1],
                analysis
            )
            
            logging.info(f"Анализ сектора {self.current_position}: "
                        f"пол={analysis.get('black_status')}, "
                        f"green={analysis.get('has_green')}, "
                        f"blue={analysis.get('has_blue_oval')}, "
                        f"red={analysis.get('has_red_oval')}")
        
        # Строим ребра графа
        self.graph.build_edges()
    
    def select_next_target(self) -> Optional[Tuple[int, int]]:
        """
        Выбрать следующую целевую точку.
        
        Returns:
            Tuple[int, int] или None
        """
        # Проверяем, исследованно ли поле
        if self.graph.is_field_explored(
            self.config.REQUIRED_GREEN_PIPES,
            self.config.REQUIRED_COLORED_PIPES
        ):
            logging.info("=== ПОЛЕ ИССЛЕДОВАНО ===")
            return None
        
        # Находим ближайший непосещенный сектор
        target = self.graph.get_nearest_unvisited(self.current_position)
        
        if target:
            # Находим путь
            distance, commands, path = self.graph.find_path_to(
                self.current_position, target
            )
            
            self.pending_commands = list(commands)  # Копируем команды
            
            logging.info(f"Выбрана цель: {target}, путь: {commands}")
            
            # Убираем флаг перекрестка - мы начинаем движение
            self.is_on_crossroad = False
        
        return target
    
    def navigate_to_target(self, unwrapped: np.ndarray):
        """
        Навигация к целевой точке.
        
        Args:
            unwrapped: Unwrapped изображение
        """
        # Определяем направление к цели
        if self.target_sector is None:
            return
        
        h, w = unwrapped.shape[:2]
        center = (w // 2, h // 2)
        
        # Получаем координаты цели в пикселях
        target_x, target_y = self.graph.get_sector_center(
            self.target_sector[0], self.target_sector[1]
        )
        
        # Вычисляем ошибку положения
        error_x = target_x - center[0]
        error_y = target_y - center[1]
        
        distance = np.sqrt(error_x**2 + error_y**2)
        angle = np.degrees(np.arctan2(error_y, error_x))
        
        # Проверяем, достигли ли цели
        if distance < self.config.CROSSROAD_DETECT_THRESHOLD:
            self.on_reached_target()
            return
        
        # Проверяем, приближаемся ли к перекрестку
        approaching = self.crossroad_detector.is_approaching_crossroad(unwrapped)
        
        if self.motion.should_stop_at_crossroad(approaching):
            # Начинаем торможение
            logging.info("Торможение перед перекрестком")
        
        # Обновляем движение
        is_moving, left_speed, right_speed = self.motion.update(33)  # ~30fps
        
        if not is_moving and self.is_on_crossroad:
            # Мы остановились на перекрестке
            return
        
        # Корректируем скорости на основе ошибки
        # Пропорциональный регулятор
        kp_angle = 5
        correction = int(kp_angle * angle / 90.0)
        correction = constrain(correction, -200, 200)
        
        # Применяем коррекцию
        left = constrain(left_speed + correction, 0, self.config.MAX_SPEED)
        right = constrain(right_speed - correction, 0, self.config.MAX_SPEED)
        
        # Для поворота на месте
        if abs(angle) > 30:
            # Нужно поворачивать
            left = self.config.MAX_SPEED // 2 if angle > 0 else -self.config.MAX_SPEED // 2
            right = -left
        
        self.motor_speeds = [left, left, right, right]
    
    def on_reached_target(self):
        """
        Обработка достижения целевой точки.
        """
        logging.info(f"Достигнута цель: {self.target_sector}")
        
        # Обновляем позицию
        self.current_position = self.target_sector
        self.graph.set_current_position(self.current_position[0], 
                                       self.current_position[1],
                                       self.current_direction)
        
        # Останавливаем движение
        self.motion.stop()
        self.target_sector = None
        self.is_on_crossroad = True
        
        # Сбрасываем ошибку localization
        if self.localization:
            self.localization.reset()
    
    def process_crossroad(self, unwrapped: np.ndarray):
        """
        Обработка перекрестка.
        
        Args:
            unwrapped: Unwrapped изображение
        """
        # Если есть команды движения
        if self.pending_commands:
            cmd = self.pending_commands.pop(0)
            
            if cmd in ('R', 'L', 'A'):
                # Нужен поворот
                self.motion.request_turn_stop()
                # TODO: Выполнить поворот через ESP
                # R = правый поворот, L = левый, A = разворот
                self._execute_turn(cmd)
                
                # Обновляем направление
                self._update_direction(cmd)
                
            elif cmd.startswith('F'):
                # Движение вперед на N сегментов
                # TODO: Можно оптимизировать - не останавливаться между сегментами
                steps = int(cmd[1:]) if len(cmd) > 1 else 1
                logging.info(f"Движение вперед: {steps} сегментов")
                # Команды добавлены обратно с уменьшенным количеством
                if steps > 1:
                    self.pending_commands.insert(0, f"F{steps-1}")
                
            self.is_on_crossroad = False  # Начинаем движение
        else:
            # Нет команд - выбираем новую цель
            self.select_next_target()
    
    def _execute_turn(self, turn_cmd: str):
        """
        Выполнить поворот.
        
        Args:
            turn_cmd: 'R', 'L' или 'A'
        """
        # Формируем команды моторов для поворота
        if turn_cmd == 'R':
            # Правый поворот: левые вперед, правые назад
            self.motor_speeds = [self.config.MAX_SPEED, self.config.MAX_SPEED,
                                -self.config.MAX_SPEED, -self.config.MAX_SPEED]
        elif turn_cmd == 'L':
            # Левый поворот: правые вперед, левые назад
            self.motor_speeds = [-self.config.MAX_SPEED, -self.config.MAX_SPEED,
                                self.config.MAX_SPEED, self.config.MAX_SPEED]
        elif turn_cmd == 'A':
            # Разворот
            self.motor_speeds = [self.config.MAX_SPEED, self.config.MAX_SPEED,
                                -self.config.MAX_SPEED, -self.config.MAX_SPEED]
        
        logging.info(f"Поворот {turn_cmd}: моторы={self.motor_speeds}")
    
    def _update_direction(self, turn_cmd: str):
        """
        Обновить направление после поворота.
        
        Args:
            turn_cmd: 'R', 'L' или 'A'
        """
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
    
    def send_command_to_esp(self):
        """
        Отправить команду на ESP.
        Вызывается один раз за цикл.
        """
        # Формируем команду
        speeds = self.motor_speeds
        servos = [0, 0, 0, 0]  # Пока без серво
        
        self.esp.sendMotionCommand(speeds, servos)
        
        logging.debug(f"ESP <- speeds={speeds}")
    
    def main_loop(self):
        """
        Главный цикл системы.
        """
        logging.info("=== НАЧАЛО ГЛАВНОГО ЦИКЛА ===")
        
        cycle_count = 0
        
        try:
            while True:
                cycle_start = time.time()
                cycle_count += 1
                
                # Захватываем кадр
                frame = self.capture_frame()
                if frame is None:
                    logging.warning("Не удалось захватить кадр")
                    time.sleep(0.1)
                    continue
                
                # Инициализируем remap при первом кадре
                if self.map_x is None:
                    self.initialize_remap(frame)
                
                # Разворачиваем в top-down
                unwrapped = self.unwrap_frame(frame)
                
                # Проверяем, на перекрестке ли мы
                is_crossroad = self.crossroad_detector.is_on_crossroad(unwrapped)
                
                # Обновляем позицию через localization
                if self.localization:
                    self.localization.unwrap_frame(frame)
                    loc_x, loc_y, loc_rot, _ = self.localization.track_displacement(unwrapped)
                
                # --- ОСНОВНАЯ ЛОГИКА ---
                
                if self.is_on_crossroad:
                    # Мы на перекрестке
                    
                    if is_crossroad:
                        # Все еще на перекрестке
                        # Проверяем, есть ли команды движения
                        if not self.pending_commands:
                            # Обрабатываем перекресток
                            self.process_crossroad(unwrapped)
                        else:
                            # Продолжаем выполнять команды
                            self.process_crossroad(unwrapped)
                    else:
                        # Ушли с перекрестка
                        self.is_on_crossroad = False
                        logging.info("Покидаем перекресток")
                
                else:
                    # Мы в движении к цели
                    
                    if is_crossroad:
                        # Достигли перекрестка
                        # Проверяем, нужен ли поворот
                        if self.pending_commands:
                            cmd = self.pending_commands[0]
                            if cmd in ('R', 'L', 'A'):
                                # Нужен поворот - останавливаемся
                                self.is_on_crossroad = True
                                logging.info("Остановка на перекрестке для поворота")
                                self.motion.stop()
                            # else: можно проезжать без остановки
                        else:
                            # Достигли цели
                            self.on_reached_target()
                    else:
                        # Продолжаем движение
                        self.navigate_to_target(unwrapped)
                
                # --- ОТПРАВКА КОМАНДЫ ---
                self.send_command_to_esp()
                
                # --- ВИЗУАЛИЗАЦИЯ ---
                status = self._get_status_string()
                self.visualizer.show(unwrapped, status)
                
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
    
    def _get_status_string(self) -> str:
        """
        Получить строку статуса для отображения.
        
        Returns:
            str: Текст статуса
        """
        status = self.graph.get_status()
        
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
            f"Исследован: {status['is_explored']}"
        ]
        
        return '\n'.join(lines)
    
    def shutdown(self):
        """
        Корректное завершение работы.
        """
        logging.info("=== ЗАВЕРШЕНИЕ РАБОТЫ ===")
        
        # Останавливаем моторы
        self.motor_speeds = [0, 0, 0, 0]
        self.send_command_to_esp()
        
        # Закрываем соединение с ESP
        self.esp.close()
        
        # Закрываем видео
        if self.cap:
            self.cap.release()
        
        # Закрываем визуализацию
        self.visualizer.close()
        
        # Финальный статус
        status = self.graph.get_status()
        logging.info(f"Финальный статус: {status}")
        
        print("\n" + "="*50)
        print("ИССЛЕДОВАНИЕ ЗАВЕРШЕНО")
        print("="*50)
        print(f"Исследованно секторов: {status['visited_sectors']}/{status['total_sectors']}")
        print(f"Найдено зеленых труб: {status['green_pipes']}")
        print(f"Найдено синих труб: {status['blue_pipes']}")
        print(f"Найдено красных труб: {status['red_pipes']}")
        print(f"Найдено пандусов: {status['ramps']}")
        print("="*50)


# =============================================================================
# ТОЧКА ВХОДА
# =============================================================================

def main():
    """
    Главная функция.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='SLAM Controller')
    parser.add_argument('--video', '-v', type=str, default=None,
                       help='Путь к видеофайлу для отладки')
    parser.add_argument('--camera', '-c', action='store_true',
                       help='Использовать реальную камеру')
    
    args = parser.parse_args()
    
    # Создаем и запускаем контроллер
    controller = SLAMController(
        video_path=args.video,
        use_camera=args.camera
    )
    
    controller.main_loop()


if __name__ == "__main__":
    main()