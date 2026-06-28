#!/usr/bin/env python3
"""
Модуль управления роботом.

Тележка:
- 4 мотора (скорости -32768..32767): [motor_fl, motor_bl, motor_fr, motor_br]
  - motor_fl, motor_bl - левые моторы (передний/задний)
  - motor_fr, motor_br - правые моторы (передний/задний)
- 4 сервомотора (положения 0..65535):
  - servo[0], servo[1] - управляют поворотом колес (для поворота налево/направо)
  - servo[2], servo[3] - дополнительные сервомоторы

Логика движения:
1. ДВИЖЕНИЕ ВПЕРЕД: колеса ровно, моторы вращаются с одинаковой скоростью
2. ПОВОРОТ: 
   a. ОСТАНОВКА на CROSSROAD_STOP_TIME_MS
   b. ПОВОРОТ КОЛЕС: сервомоторы 0,1 -> TURN_ANGLE
   c. ПОВОРОТ ТЕЛЕЖКИ: моторы вращаются в противоположных направлениях,
      контроль по тикам энкодера (используем TICK_PER_ONE_90_TURN)
   d. ВОЗВРАТ КОЛЕС: сервомоторы 0,1 -> STRAIGHT_ANGLE (базовое положение)

Протокол связи с ESP:
- Отправка: speeds[4] (int16) + servos[4] (uint16)
- Получение: mode + encoder_ticks[4] (int32)
"""

import logging
import time
from enum import Enum, auto
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

# =============================================================================
# КОНФИГУРАЦИЯ РОБОТА
# =============================================================================

class RobotConfig:
    """Конфигурация робота."""
    
    # --- Моторы ---
    MAX_MOTOR_SPEED = 1000      # Максимальная скорость моторов
    MIN_MOTOR_SPEED = 100      # Минимальная скорость для движения
    STOP_SPEED = 0             # Скорость остановки
    
    # --- Серво ---
    SERVO_STRAIGHT_ANGLE = 32768  # Среднее положение (колеса ровно) ~90°
    SERVO_MIN = 0                 # Минимум
    SERVO_MAX = 65535             # Максимум
    
    # Угол поворота колес для выполнения маневра
    # Примечание: реальное значение зависит от механики робота
    # Для поворота тележки на 90° колеса поворачиваются на ~45°
    TURN_ANGLE_OFFSET = 16384     # Смещение от среднего: 90° + 45° = 135°
    
    # Расчет положений серво для поворота
    SERVO_LEFT = SERVO_STRAIGHT_ANGLE + TURN_ANGLE_OFFSET   # Поворот влево
    SERVO_RIGHT = SERVO_STRAIGHT_ANGLE - TURN_ANGLE_OFFSET  # Поворот вправо
    SERVO_STRAIGHT = SERVO_STRAIGHT_ANGLE                  # Прямо
    
    # --- Тайминги ---
    CROSSROAD_STOP_TIME_MS = 500   # Время остановки на перекрестке (мс)
    SERVO_ROTATE_TIME_MS = 300     # Время поворота серво (мс)
    WHEELS_RETURN_TIME_MS = 200    # Время возврата колес в прямое положение (мс)
    
    # --- Движение ---
    # Трапецеидальный профиль скорости
    ACCEL_TIME_MS = 300            # Время разгона (мс)
    CRUISE_TIME_MS = 500          # Время движения с максимальной скоростью (мс)
    DECEL_TIME_MS = 300           # Время торможения (мс)
    
    # --- Поворот тележки ---
    # Сколько тиков энкодера на 90° поворот
    # ЭТО ЗНАЧЕНИЕ НУЖНО ОПРЕДЕЛИТЬ ЭКСПЕРИМЕНТАЛЬНО!
    TICK_PER_90_TURN = 1000        # Тиков на 90° поворот
    TURN_MOTOR_SPEED = 400         # Скорость моторов при повороте тележки
    
    # --- Тики на единицу пути ---
    # Для движения вперед - можно использовать для точной остановки
    TICKS_PER_CROSSROAD = 2000     # Примерное количество тиков между перекрестками
    
    # --- ESP коммуникация ---
    RESPONSE_TIMEOUT_MS = 500      # Таймаут ожидания ответа
    MAX_MISSED_RESPONSES = 10      # Максимум пропущенных ответов


# =============================================================================
# СОСТОЯНИЯ РОБОТА
# =============================================================================

class RobotState(Enum):
    """Возможные состояния робота."""
    # Базовые состояния
    IDLE = auto()                 # Ожидание команд
    STOPPED = auto()              # Остановлен (после завершения маневра)
    
    # Движение по линии
    MOVING_FORWARD = auto()       # Движение вперед
    ACCELERATING = auto()         # Разгон
    CRUISING = auto()             # Движение с постоянной скоростью
    DECELERATING = auto()          # Торможение
    
    # Поворот на 90°
    STOPPING_FOR_TURN = auto()    # Остановка перед поворотом
    TURNING_WHEELS = auto()       # Поворот колес сервомоторами
    ROTATING_ROBOT = auto()        # Поворот тележки (моторы в противоположных направлениях)
    RETURNING_WHEELS = auto()      # Возврат колес в прямое положение
    
    # Специальные состояния
    ERROR = auto()                # Ошибка
    WAITING = auto()              # Ожидание (пауза между маневрами)


# =============================================================================
# НАПРАВЛЕНИЯ ДВИЖЕНИЯ
# =============================================================================

class Direction(Enum):
    """Направления движения робота."""
    UP = 'U'      # Вверх
    RIGHT = 'R'   # Направо
    DOWN = 'D'    # Вниз
    LEFT = 'L'    # Налево
    
    @classmethod
    def from_string(cls, s: str) -> 'Direction':
        """Создать направление из строки."""
        for d in cls:
            if d.value == s.upper():
                return d
        raise ValueError(f"Unknown direction: {s}")
    
    def turn_left(self) -> 'Direction':
        """Повернуть налево (90° CCW)."""
        turn_map = {
            Direction.UP: Direction.LEFT,
            Direction.LEFT: Direction.DOWN,
            Direction.DOWN: Direction.RIGHT,
            Direction.RIGHT: Direction.UP,
        }
        return turn_map[self]
    
    def turn_right(self) -> 'Direction':
        """Повернуть направо (90° CW)."""
        turn_map = {
            Direction.UP: Direction.RIGHT,
            Direction.RIGHT: Direction.DOWN,
            Direction.DOWN: Direction.LEFT,
            Direction.LEFT: Direction.UP,
        }
        return turn_map[self]
    
    def reverse(self) -> 'Direction':
        """Развернуться (180°)."""
        turn_map = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT,
        }
        return turn_map[self]


# =============================================================================
# ТИПЫ ПОВОРОТОВ
# =============================================================================

class TurnType(Enum):
    """Типы поворотов."""
    LEFT = 'L'      # Поворот налево (90° CCW)
    RIGHT = 'R'     # Поворот направо (90° CW)
    AROUND = 'A'    # Разворот (180°)
    NONE = ''       # Нет поворота


# =============================================================================
# ДАННЫЕ С ЭНКОДЕРОВ
# =============================================================================

@dataclass
class EncoderData:
    """Данные с энкодеров моторов."""
    ticks: List[int]          # Тики 4 моторов: [fl, bl, fr, br]
    timestamp: float           # Время получения данных
    mode: int = 0             # Режим ESP
    
    @property
    def left_ticks(self) -> int:
        """Тики левых моторов (среднее)."""
        return (self.ticks[0] + self.ticks[1]) // 2
    
    @property
    def right_ticks(self) -> int:
        """Тики правых моторов (среднее)."""
        return (self.ticks[2] + self.ticks[3]) // 2
    
    @property
    def front_left_ticks(self) -> int:
        return self.ticks[0]
    
    @property
    def back_left_ticks(self) -> int:
        return self.ticks[1]
    
    @property
    def front_right_ticks(self) -> int:
        return self.ticks[2]
    
    @property
    def back_right_ticks(self) -> int:
        return self.ticks[3]


# =============================================================================
# КОМАНДЫ УПРАВЛЕНИЯ
# =============================================================================

@dataclass
class MotionCommand:
    """Команда движения."""
    command_type: str           # 'F' - forward, 'L' - left, 'R' - right, 'A' - around
    segments: int = 1          # Количество сегментов (для F)
    
    @classmethod
    def forward(cls, segments: int = 1) -> 'MotionCommand':
        return cls('F', segments)
    
    @classmethod
    def left(cls) -> 'MotionCommand':
        return cls('L', 1)
    
    @classmethod
    def right(cls) -> 'MotionCommand':
        return cls('R', 1)
    
    @classmethod
    def around(cls) -> 'MotionCommand':
        return cls('A', 1)


# =============================================================================
# ОСНОВНОЙ КЛАСС УПРАВЛЕНИЯ РОБОТОМ
# =============================================================================

class RobotController:
    """
    Контроллер управления роботом.
    
    Управляет:
    - Движением вперед по линии с трапецеидальным профилем скорости
    - Поворотами на 90°/180° с подсчетом тиков энкодера
    - Серводвигателями для поворота колес
    - Очередью команд движения
    """
    
    def __init__(self, esp_comm, config: type = RobotConfig, debug: bool = True):
        """
        Инициализация контроллера.
        
        Args:
            esp_comm: Объект ESPCommunication для отправки команд
            config: Класс конфигурации
            debug: Включить отладочный вывод
        """
        self.esp = esp_comm
        self.config = config
        self.debug = debug
        
        # Состояние робота
        self.state = RobotState.IDLE
        self.direction = Direction.UP  # Начальное направление
        
        # Очередь команд
        self.command_queue: List[MotionCommand] = []
        
        # Текущие скорости моторов и положения серво
        self.motor_speeds: List[int] = [0, 0, 0, 0]  # [fl, bl, fr, br]
        self.servo_positions: List[int] = [config.SERVO_STRAIGHT] * 4
        
        # Данные энкодеров
        self.last_encoder_data: Optional[EncoderData] = None
        self.encoder_at_turn_start: Optional[EncoderData] = None
        
        # Таймеры состояний
        self.state_start_time: float = time.time() * 1000
        self.current_time_ms: float = 0
        
        # Траектория движения (для отладки)
        self.position = (0, 0)  # (row, col)
        
        # Статистика
        self.command_count = 0
        self.error_count = 0
        
        # Инициализация логирования
        self._log("RobotController initialized")
    
    def _log(self, message: str, level: str = "INFO"):
        """Логирование сообщения."""
        if self.debug:
            print(f"[ROBOT][{level}] {message}")
        logging.info(f"[ROBOT] {message}")
    
    def _debug(self, message: str):
        """Отладочное сообщение."""
        if self.debug:
            self._log(message, "DEBUG")
    
    def _error(self, message: str):
        """Сообщение об ошибке."""
        self._log(message, "ERROR")
        self.error_count += 1
    
    def _get_time_ms(self) -> float:
        """Получить текущее время в миллисекундах."""
        return time.time() * 1000
    
    # =========================================================================
    # УПРАВЛЕНИЕ КОМАНДАМИ
    # =========================================================================
    
    def set_commands(self, commands: List[str]):
        """
        Установить очередь команд.
        
        Args:
            commands: Список команд в строковом формате
                      ('F', 'L', 'R', 'A', 'F2', 'F3', etc.)
        """
        self.command_queue = []
        for cmd_str in commands:
            cmd_str = cmd_str.strip().upper()
            if not cmd_str:
                continue
            
            if cmd_str[0] == 'F':
                # Forward с количеством сегментов
                segments = int(cmd_str[1:]) if len(cmd_str) > 1 else 1
                self.command_queue.append(MotionCommand.forward(segments))
            elif cmd_str[0] in ('L', 'R', 'A'):
                # Поворот
                self.command_queue.append(MotionCommand(cmd_str[0]))
        
        self._log(f"Commands set: {[str(c.command_type) + str(c.segments) if c.segments > 1 else c.command_type for c in self.command_queue]}")
    
    def add_command(self, command: MotionCommand):
        """Добавить команду в очередь."""
        self.command_queue.append(command)
        self._debug(f"Command added: {command.command_type}")
    
    def clear_commands(self):
        """Очистить очередь команд."""
        self.command_queue.clear()
        self._debug("Commands cleared")
    
    def get_next_command(self) -> Optional[MotionCommand]:
        """Получить следующую команду из очереди."""
        if self.command_queue:
            return self.command_queue[0]
        return None
    
    def pop_command(self) -> Optional[MotionCommand]:
        """Извлечь следующую команду из очереди."""
        if self.command_queue:
            cmd = self.command_queue.pop(0)
            self.command_count += 1
            self._debug(f"Command popped: {cmd.command_type}, remaining: {len(self.command_queue)}")
            return cmd
        return None
    
    def peek_command(self) -> Optional[MotionCommand]:
        """Посмотреть следующую команду без извлечения."""
        if self.command_queue:
            return self.command_queue[0]
        return None
    
    def has_commands(self) -> bool:
        """Проверить наличие команд в очереди."""
        return len(self.command_queue) > 0
    
    # =========================================================================
    # ОТПРАВКА КОМАНД НА ESP
    # =========================================================================
    
    def send_command(self) -> bool:
        """
        Отправить текущую команду на ESP и получить ответ.
        
        Returns:
            bool: True если ответ получен успешно
        """
        success, response = self.esp.sendMotionCommand(
            self.motor_speeds,
            self.servo_positions
        )
        
        if success:
            # Парсим данные энкодера из ответа
            values = response.get('values', [0, 0, 0, 0])
            self.last_encoder_data = EncoderData(
                ticks=values,
                timestamp=self._get_time_ms(),
                mode=response.get('mode', 0)
            )
            self._debug(f"ESP response: ticks={[v for v in values]}")
        else:
            self._error(f"ESP no response! speeds={self.motor_speeds}")
        
        return success
    
    # =========================================================================
    # УПРАВЛЕНИЕ СОСТОЯНИЯМИ
    # =========================================================================
    
    def set_state(self, new_state: RobotState):
        """Сменить состояние робота."""
        old_state = self.state
        self.state = new_state
        self.state_start_time = self._get_time_ms()
        self._log(f"State: {old_state.name} -> {new_state.name}")
    
    def get_state_duration_ms(self) -> float:
        """Получить время в текущем состоянии (мс)."""
        return self._get_time_ms() - self.state_start_time
    
    def is_moving(self) -> bool:
        """Проверить, движется ли робот."""
        return self.state not in (RobotState.IDLE, RobotState.STOPPED, RobotState.WAITING, RobotState.ERROR)
    
    def is_turning(self) -> bool:
        """Проверить, выполняется ли поворот."""
        return self.state in (
            RobotState.STOPPING_FOR_TURN,
            RobotState.TURNING_WHEELS,
            RobotState.ROTATING_ROBOT,
            RobotState.RETURNING_WHEELS
        )
    
    # =========================================================================
    # УПРАВЛЕНИЕ МОТОРАМИ
    # =========================================================================
    
    def set_motor_speeds(self, left: int, right: int):
        """
        Установить скорости моторов.
        
        Args:
            left: Скорость левых моторов (всегда одинаковая для fl и bl)
            right: Скорость правых моторов (всегда одинаковая для fr и br)
        """
        self.motor_speeds = [left, left, right, right]
        self._debug(f"Motors set: left={left}, right={right}")
    
    def stop_motors(self):
        """Остановить все моторы."""
        self.set_motor_speeds(0, 0)
    
    def set_turning_speeds(self, turn_type: TurnType):
        """
        Установить скорости для поворота тележки.
        
        Args:
            turn_type: Тип поворота (LEFT - колеса влево, RIGHT - колеса вправо)
        """
        speed = self.config.TURN_MOTOR_SPEED
        
        if turn_type == TurnType.LEFT:
            # Левые моторы назад, правые вперед -> поворот налево
            self.set_motor_speeds(-speed, speed)
        elif turn_type == TurnType.RIGHT:
            # Левые моторы вперед, правые назад -> поворот направо
            self.set_motor_speeds(speed, -speed)
        elif turn_type == TurnType.AROUND:
            # Разворот - один полный оборот
            self.set_motor_speeds(speed, -speed)
    
    # =========================================================================
    # УПРАВЛЕНИЕ СЕРВОМОТОРАМИ
    # =========================================================================
    
    def set_servos_straight(self):
        """Установить колеса прямо."""
        self.servo_positions = [self.config.SERVO_STRAIGHT] * 4
        self._debug("Servos: STRAIGHT")
    
    def set_servos_for_turn(self, turn_type: TurnType):
        """
        Установить положение серво для поворота.
        
        ВСЕ 4 колеса поворачиваются в одну сторону для выполнения поворота тележки.
        
        Args:
            turn_type: Тип поворота
        """
        if turn_type == TurnType.LEFT:
            # Все колеса поворачиваются влево
            self.servo_positions = [
                self.config.SERVO_LEFT,   # Передний левый - влево
                self.config.SERVO_LEFT,   # Задний левый - влево
                self.config.SERVO_LEFT,   # Передний правый - влево
                self.config.SERVO_LEFT    # Задний правый - влево
            ]
        elif turn_type == TurnType.RIGHT:
            # Все колеса поворачиваются вправо
            self.servo_positions = [
                self.config.SERVO_RIGHT,  # Передний левый - вправо
                self.config.SERVO_RIGHT,  # Задний левый - вправо
                self.config.SERVO_RIGHT,  # Передний правый - вправо
                self.config.SERVO_RIGHT   # Задний правый - вправо
            ]
        elif turn_type == TurnType.AROUND:
            # Для разворота можно использовать любое положение
            # Используем RIGHT для определенности
            self.servo_positions = [
                self.config.SERVO_RIGHT,
                self.config.SERVO_RIGHT,
                self.config.SERVO_RIGHT,
                self.config.SERVO_RIGHT
            ]
        
        self._debug(f"Servos set for turn: {turn_type.name}")
    
    # =========================================================================
    # ДВИЖЕНИЕ ВПЕРЕД С ТРАПЕЦЕИДАЛЬНЫМ ПРОФИЛЕМ
    # =========================================================================
    
    def start_forward_motion(self):
        """Начать движение вперед с разгоном."""
        self.set_state(RobotState.ACCELERATING)
        self.set_motor_speeds(self.config.MIN_MOTOR_SPEED, self.config.MIN_MOTOR_SPEED)
        self.set_servos_straight()
    
    def update_forward_motion(self) -> bool:
        """
        Обновить состояние движения вперед.
        
        Returns:
            bool: True если движение продолжается, False если завершено
        """
        elapsed = self.get_state_duration_ms()
        
        if self.state == RobotState.ACCELERATING:
            # Разгон
            if elapsed >= self.config.ACCEL_TIME_MS:
                self.set_state(RobotState.CRUISING)
                self.set_motor_speeds(self.config.MAX_MOTOR_SPEED, self.config.MAX_MOTOR_SPEED)
            else:
                # Линейная интерполяция скорости
                progress = elapsed / self.config.ACCEL_TIME_MS
                speed = int(
                    self.config.MIN_MOTOR_SPEED + 
                    (self.config.MAX_MOTOR_SPEED - self.config.MIN_MOTOR_SPEED) * progress
                )
                self.set_motor_speeds(speed, speed)
        
        elif self.state == RobotState.CRUISING:
            # Крейсерская скорость
            if elapsed >= self.config.CRUISE_TIME_MS:
                self.set_state(RobotState.DECELERATING)
            else:
                self.set_motor_speeds(self.config.MAX_MOTOR_SPEED, self.config.MAX_MOTOR_SPEED)
        
        elif self.state == RobotState.DECELERATING:
            # Торможение
            if elapsed >= self.config.DECEL_TIME_MS:
                self.stop_motors()
                self.set_state(RobotState.STOPPED)
                return False
            else:
                # Линейная интерполяция скорости
                progress = elapsed / self.config.DECEL_TIME_MS
                speed = int(
                    self.config.MIN_MOTOR_SPEED + 
                    (self.config.MAX_MOTOR_SPEED - self.config.MIN_MOTOR_SPEED) * (1 - progress)
                )
                self.set_motor_speeds(speed, speed)
        
        return True
    
    # =========================================================================
    # ПОВОРОТ ТЕЛЕЖКИ
    # =========================================================================
    
    def start_turn(self, turn_type: TurnType):
        """
        Начать поворот.
        
        Args:
            turn_type: Тип поворота (LEFT, RIGHT, AROUND)
        """
        self._log(f"Starting turn: {turn_type.name}")
        
        # Запоминаем начальные тики для отслеживания поворота
        if self.last_encoder_data:
            self.encoder_at_turn_start = EncoderData(
                ticks=self.last_encoder_data.ticks.copy(),
                timestamp=self._get_time_ms()
            )
        else:
            self.encoder_at_turn_start = EncoderData(
                ticks=[0, 0, 0, 0],
                timestamp=self._get_time_ms()
            )
        
        # Этап 1: Остановка
        self.set_state(RobotState.STOPPING_FOR_TURN)
        self.stop_motors()
        self.set_servos_straight()
        
        # Запоминаем тип поворота для последующих этапов
        self._current_turn_type = turn_type
    
    def update_turn(self) -> bool:
        """
        Обновить состояние поворота.
        
        Returns:
            bool: True если поворот продолжается, False если завершен
        """
        elapsed = self.get_state_duration_ms()
        turn_type = getattr(self, '_current_turn_type', TurnType.RIGHT)
        
        if self.state == RobotState.STOPPING_FOR_TURN:
            # Этап 1: Ожидание остановки
            if elapsed >= self.config.CROSSROAD_STOP_TIME_MS:
                self.set_state(RobotState.TURNING_WHEELS)
                self.set_servos_for_turn(turn_type)
        
        elif self.state == RobotState.TURNING_WHEELS:
            # Этап 2: Поворот колес сервомоторами
            if elapsed >= self.config.SERVO_ROTATE_TIME_MS:
                self.set_state(RobotState.ROTATING_ROBOT)
                self.set_turning_speeds(turn_type)
                # Сбрасываем таймер для отслеживания тиков
                self.state_start_time = self._get_time_ms()
        
        elif self.state == RobotState.ROTATING_ROBOT:
            # Этап 3: Поворот тележки с подсчетом тиков
            if not self.last_encoder_data:
                self._debug("Waiting for encoder data...")
                return True
            
            # Подсчет тиков с момента начала поворота
            start_ticks = self.encoder_at_turn_start.ticks
            current_ticks = self.last_encoder_data.ticks
            
            # Разница тиков (абсолютное значение)
            delta_left = abs(current_ticks[0] - start_ticks[0])
            delta_right = abs(current_ticks[2] - start_ticks[2])
            avg_delta = (delta_left + delta_right) // 2
            
            self._debug(f"Turn progress: avg_delta={avg_delta}, target={self.config.TICK_PER_90_TURN}")
            
            # Определяем количество 90° поворотов для завершения
            if turn_type == TurnType.AROUND:
                target_ticks = self.config.TICK_PER_90_TURN * 2  # 180° = 2 * 90°
            else:
                target_ticks = self.config.TICK_PER_90_TURN  # 90°
            
            if avg_delta >= target_ticks:
                self.stop_motors()
                self.set_state(RobotState.RETURNING_WHEELS)
                self.set_servos_straight()
        
        elif self.state == RobotState.RETURNING_WHEELS:
            # Этап 4: Возврат колес в прямое положение
            if elapsed >= self.config.WHEELS_RETURN_TIME_MS:
                self.set_state(RobotState.STOPPED)
                
                # Обновляем направление
                self._update_direction(turn_type)
                
                self._log(f"Turn complete! New direction: {self.direction.value}")
                return False
        
        return True
    
    def _update_direction(self, turn_type: TurnType):
        """Обновить направление после поворота."""
        if turn_type == TurnType.LEFT:
            self.direction = self.direction.turn_left()
        elif turn_type == TurnType.RIGHT:
            self.direction = self.direction.turn_right()
        elif turn_type == TurnType.AROUND:
            self.direction = self.direction.reverse()
    
    # =========================================================================
    # ГЛАВНЫЙ ЦИКЛ УПРАВЛЕНИЯ
    # =========================================================================
    
    def update(self) -> bool:
        """
        Главный метод обновления состояния робота.
        Вызывается каждый цикл.
        
        Returns:
            bool: True если робот в рабочем состоянии, False если есть ошибка
        """
        # Проверка соединения
        if not self.esp.is_connected():
            self._error("ESP connection lost!")
            self.set_state(RobotState.ERROR)
            return False
        
        # Обработка состояний
        if self.state == RobotState.IDLE:
            # Проверяем, есть ли команды
            if self.has_commands():
                cmd = self.peek_command()
                if cmd.command_type == 'F':
                    # Движение вперед
                    self.start_forward_motion()
                elif cmd.command_type in ('L', 'R', 'A'):
                    # Поворот
                    turn_type = TurnType(cmd.command_type)
                    self.start_turn(turn_type)
        
        elif self.state in (RobotState.ACCELERATING, RobotState.CRUISING, RobotState.DECELERATING):
            # Движение вперед
            if not self.update_forward_motion():
                # Движение завершено
                cmd = self.get_next_command()
                if cmd and cmd.command_type == 'F':
                    # Уменьшаем счетчик сегментов
                    if cmd.segments > 1:
                        cmd.segments -= 1
                    else:
                        self.pop_command()  # Убираем выполненную команду
                    
                    # Если еще есть сегменты или команды
                    if cmd.segments > 1:
                        self.start_forward_motion()
                    elif self.has_commands():
                        # Ждем следующую команду
                        self.set_state(RobotState.WAITING)
                    else:
                        self.set_state(RobotState.IDLE)
        
        elif self.state in (
            RobotState.STOPPING_FOR_TURN,
            RobotState.TURNING_WHEELS,
            RobotState.ROTATING_ROBOT,
            RobotState.RETURNING_WHEELS
        ):
            # Поворот
            if not self.update_turn():
                # Поворот завершен
                self.pop_command()
                
                if self.has_commands():
                    # Ждем следующую команду
                    self.set_state(RobotState.WAITING)
                else:
                    self.set_state(RobotState.IDLE)
        
        elif self.state == RobotState.WAITING:
            # Ожидание между командами
            if self.has_commands():
                cmd = self.peek_command()
                if cmd.command_type == 'F':
                    self.start_forward_motion()
                elif cmd.command_type in ('L', 'R', 'A'):
                    turn_type = TurnType(cmd.command_type)
                    self.start_turn(turn_type)
        
        elif self.state == RobotState.STOPPED:
            # Остановлен, проверяем команды
            if self.has_commands():
                cmd = self.peek_command()
                if cmd.command_type == 'F':
                    self.start_forward_motion()
                elif cmd.command_type in ('L', 'R', 'A'):
                    turn_type = TurnType(cmd.command_type)
                    self.start_turn(turn_type)
        
        # Отправляем команду на ESP
        return self.send_command()
    
    def reset(self):
        """Сбросить контроллер в начальное состояние."""
        self.state = RobotState.IDLE
        self.command_queue.clear()
        self.stop_motors()
        self.set_servos_straight()
        self.direction = Direction.UP
        self.position = (0, 0)
        self._log("RobotController reset")
    
    # =========================================================================
    # ИНФОРМАЦИЯ О СОСТОЯНИИ
    # =========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Получить статус робота."""
        return {
            'state': self.state.name,
            'direction': self.direction.value,
            'position': self.position,
            'motors': self.motor_speeds,
            'servos': self.servo_positions,
            'commands_in_queue': len(self.command_queue),
            'command_count': self.command_count,
            'error_count': self.error_count,
            'is_moving': self.is_moving(),
            'is_turning': self.is_turning(),
            'last_encoder': self.last_encoder_data.ticks if self.last_encoder_data else None
        }
    
    def get_commands_string(self) -> str:
        """Получить строку с текущими командами."""
        return ''.join([
            (c.command_type + str(c.segments) if c.segments > 1 else c.command_type)
            for c in self.command_queue
        ])
    
    def __str__(self) -> str:
        """Строковое представление состояния."""
        status = self.get_status()
        return (
            f"Robot: state={status['state']}, dir={status['direction']}, "
            f"pos={status['position']}, motors={status['motors']}, "
            f"cmds={self.get_commands_string()}"
        )


# =============================================================================
# ТЕСТОВЫЙ РЕЖИМ
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s'
    )
    
    print("=" * 60)
    print("Robot Control Module - Test Mode")
    print("=" * 60)
    
    # Создаем mock ESP для тестирования
    class MockESP:
        def __init__(self):
            self.ticks = [0, 0, 0, 0]
            self.connected = True
            self.sent_count = 0
        
        def is_connected(self):
            return self.connected
        
        def sendMotionCommand(self, speeds, servos):
            # Симулируем ответ
            for i in range(4):
                if speeds[i] > 0:
                    self.ticks[i] += 10
                elif speeds[i] < 0:
                    self.ticks[i] -= 10
            
            self.sent_count += 1
            return True, {
                'mode': 0,
                'values': self.ticks.copy()
            }
    
    # Тест
    mock_esp = MockESP()
    robot = RobotController(mock_esp, debug=True)
    
    # Тест движения вперед
    print("\n--- Test 1: Forward Motion ---")
    robot.set_commands(['F'])
    while robot.state != RobotState.STOPPED:
        robot.update()
        print(f"  {robot}")
    
    # Тест поворота направо
    print("\n--- Test 2: Right Turn ---")
    robot.reset()
    robot.set_commands(['R'])
    while robot.state != RobotState.STOPPED:
        robot.update()
        print(f"  {robot}")
    
    # Тест серии команд
    print("\n--- Test 3: Command Sequence ---")
    robot.reset()
    robot.set_commands(['F', 'R', 'F2', 'L', 'A'])
    
    print(f"Commands: {robot.get_commands_string()}")
    print(f"Status: {robot.get_status()}")
    
    print("\n" + "=" * 60)
    print("Tests completed!")
    print("=" * 60)