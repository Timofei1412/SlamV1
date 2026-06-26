#!/usr/bin/env python3
"""
Модуль построения графа исследованного пространства.

Логика:
- Каждый сектор (ячейка сетки) характеризуется уровнем пола (белый/черный)
- Ребро между секторами можно построить если:
  1. Следующий сектор имеет тот же уровень пола И нет синих/красных/зеленых труб
  2. Исключение: пандус - темная фигура с кругами синего и красного цвета
     - Можно использовать если: красный слева, синий справа и текущий уровень белый
     - Или: синий слева, красный справа и текущий уровень черный
- Синие, красные и зеленые трубы заносятся в отдельные массивы для маршрутизации
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from enum import Enum
import logging

from router import Pathfinder


class FloorLevel(Enum):
    """Уровень пола"""
    WHITE = "white"
    BLACK = "black"
    UNKNOWN = "unknown"


@dataclass
class Pipe:
    """Труба - препятствие на секторе"""
    sector: Tuple[int, int]  # (row, col) сектора
    color: str  # "blue", "red", "green"
    position: Tuple[float, float]  # (x, y) позиция в пикселях на unwrapped image


@dataclass
class Ramp:
    """Пандус - переход между уровнями"""
    sector: Tuple[int, int]
    direction_from: str  # направление входа: 'U', 'D', 'L', 'R'
    target_level: str  # "white" или "black" - уровень после пандуса
    # Позиции кругов относительно центра пандуса
    circles: List[Tuple[str, float, float]]  # [(color, x_offset, y_offset), ...]


@dataclass
class Sector:
    """Сектор (ячейка сетки)"""
    row: int
    col: int
    floor_level: FloorLevel = FloorLevel.UNKNOWN
    has_green_pipe: bool = False
    pipes: List[Pipe] = field(default_factory=list)
    ramp: Optional[Ramp] = None
    visited: bool = False
    
    def __hash__(self):
        return hash((self.row, self.col))


class BuildGraph:
    """
    Класс для построения и управления графом исследованного пространства.
    
    API:
        add_sector_analysis(row, col, analysis_result) - добавить анализ сектора
        get_edge_possible(current_pos, direction) -> bool - проверить возможность перехода
        build_edges() - построить все возможные ребра в графе
        get_pipe_positions(color) -> List[Pipe] - получить позиции труб определенного цвета
        get_unvisited_sectors() -> List[Tuple[int, int]] - получить непосещенные секторы
        get_nearest_unvisited(start) -> Optional[Tuple[int, int]] - ближайший непосещенный сектор
        find_path_to(start, end) -> Tuple[int, str, List[Tuple]] - найти путь
        visualize() - визуализировать граф
    """
    
    # Пороги для определения уровня пола (из analyse.py)
    BLACK_AREA_THRESHOLD = 10000
    
    def __init__(self, grid_rows: int = 4, grid_cols: int = 4):
        """
        Инициализация графа.
        
        Args:
            grid_rows: Количество строк в сетке
            grid_cols: Количество столбцов в сетке
        """
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        
        # Секторы: ключ = (row, col)
        self.sectors: Dict[Tuple[int, int], Sector] = {}
        
        # Трубы по цветам
        self.green_pipes: List[Pipe] = []
        self.blue_pipes: List[Pipe] = []
        self.red_pipes: List[Pipe] = []
        
        # Пандусы
        self.ramps: List[Ramp] = []
        
        # Pathfinder для маршрутизации
        self.pathfinder = Pathfinder(max(grid_rows, grid_cols))
        
        # Инициализируем все секторы как неизвестные
        for r in range(grid_rows):
            for c in range(grid_cols):
                self.sectors[(r, c)] = Sector(row=r, col=c)
        
        # Текущая позиция и направление
        self.current_position: Tuple[int, int] = (0, 0)
        self.current_direction: str = 'U'  # U, D, L, R
        
        # Целевая точка для движения
        self.target_position: Optional[Tuple[int, int]] = None
        self.target_pixel: Optional[Tuple[float, float]] = None
        
        logging.info(f"BuildGraph инициализирован: {grid_rows}x{grid_cols}")
    
    def determine_floor_level(self, black_area_px: int) -> FloorLevel:
        """
        Определить уровень пола по площади черного цвета.
        
        Args:
            black_area_px: Площадь черного цвета в пикселях
            
        Returns:
            FloorLevel: WHITE, BLACK или UNKNOWN
        """
        if black_area_px > self.BLACK_AREA_THRESHOLD:
            return FloorLevel.BLACK
        elif black_area_px < self.BLACK_AREA_THRESHOLD // 2:
            return FloorLevel.WHITE
        else:
            return FloorLevel.UNKNOWN
    
    def detect_ramp(self, analysis: dict, sector_center_x: float, sector_center_y: float) -> Optional[Ramp]:
        """
        Определить пандус на секторе.
        
        Пандус - темная фигура с кругами синего и красного цвета.
        Условия использования:
        - Красный слева, синий справа + текущий уровень белый -> переход на черный
        - Синий слева, красный справа + текущий уровень черный -> переход на белый
        
        Args:
            analysis: Результат анализа изображения
            sector_center_x: X координата центра сектора
            sector_center_y: Y координата центра сектора
            
        Returns:
            Ramp или None
        """
        # Проверяем наличие кругов
        has_blue_oval = analysis.get('has_blue_oval', False)
        has_red_oval = analysis.get('has_red_oval', False)
        
        if not (has_blue_oval and has_red_oval):
            return None
        
        # TODO: Реализовать определение позиций кругов (слева/справа)
        # Пока возвращаем None - требуется более детальный анализ изображения
        return None
    
    def add_sector_analysis(self, row: int, col: int, analysis: dict, 
                            sector_center_x: Optional[float] = None,
                            sector_center_y: Optional[float] = None):
        """
        Добавить результат анализа сектора.
        
        Args:
            row: Строка сектора
            col: Столбец сектора
            analysis: Результат от analyse.analyze_image()
            sector_center_x: X координата центра сектора в пикселях (опционально)
            sector_center_y: Y координата центра сектора в пикселях (опционально)
        """
        if (row, col) not in self.sectors:
            logging.warning(f"Сектор ({row}, {col}) вне диапазона")
            return
        
        sector = self.sectors[(row, col)]
        
        # Определяем уровень пола
        black_area_px = analysis.get('black_area_px', 0)
        sector.floor_level = self.determine_floor_level(black_area_px)
        
        # Проверяем наличие труб
        if analysis.get('has_green', False):
            sector.has_green_pipe = True
            pipe = Pipe(
                sector=(row, col),
                color='green',
                position=(sector_center_x or 0, sector_center_y or 0)
            )
            self.green_pipes.append(pipe)
            sector.pipes.append(pipe)
        
        if analysis.get('blue_area_px', 0) > 300:
            pipe = Pipe(
                sector=(row, col),
                color='blue',
                position=(sector_center_x or 0, sector_center_y or 0)
            )
            self.blue_pipes.append(pipe)
            sector.pipes.append(pipe)
        
        if analysis.get('red_area_px', 0) > 300:
            pipe = Pipe(
                sector=(row, col),
                color='red',
                position=(sector_center_x or 0, sector_center_y or 0)
            )
            self.red_pipes.append(pipe)
            sector.pipes.append(pipe)
        
        # Проверяем пандус
        ramp = self.detect_ramp(analysis, sector_center_x, sector_center_y)
        if ramp:
            sector.ramp = ramp
            self.ramps.append(ramp)
        
        sector.visited = True
        
        logging.info(f"Сектор ({row}, {col}): ур={sector.floor_level.value}, "
                    f"green={sector.has_green_pipe}, blue={len([p for p in sector.pipes if p.color=='blue'])}, "
                    f"red={len([p for p in sector.pipes if p.color=='red'])}")
    
    def can_build_edge(self, from_pos: Tuple[int, int], direction: str) -> Tuple[bool, str]:
        """
        Проверить возможность построения ребра.
        
        Args:
            from_pos: Текущая позиция (row, col)
            direction: Направление движения ('U', 'D', 'L', 'R')
            
        Returns:
            Tuple[bool, str]: (можно_построить, причина)
        """
        dr, dc = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}[direction]
        to_row, to_col = from_pos[0] + dr, from_pos[1] + dc
        
        # Проверяем границы
        if not (0 <= to_row < self.grid_rows and 0 <= to_col < self.grid_cols):
            return False, "вне границ"
        
        from_sector = self.sectors.get(from_pos)
        to_sector = self.sectors.get((to_row, to_col))
        
        if from_sector is None or to_sector is None:
            return False, "сектор не существует"
        
        # Проверяем, исследован ли целевой сектор
        if not to_sector.visited:
            return False, "сектор не исследован"
        
        # Проверяем препятствия
        if to_sector.has_green_pipe:
            return False, "зеленая труба"
        
        if any(p.color in ('blue', 'red') for p in to_sector.pipes):
            return False, "синяя/красная труба"
        
        # Проверяем уровни пола
        if from_sector.floor_level == to_sector.floor_level:
            return True, "одинаковый уровень"
        
        # Разные уровни - проверяем пандус
        if to_sector.ramp:
            # TODO: Добавить логику проверки направления пандуса
            return True, "пандус"
        
        return False, "разные уровни без пандуса"
    
    def build_edges(self):
        """
        Построить все возможные ребра в графе на основе исследованных секторов.
        """
        edges_added = 0
        
        for (row, col), sector in self.sectors.items():
            if not sector.visited:
                continue
            
            for direction in ['U', 'D', 'L', 'R']:
                can_build, reason = self.can_build_edge((row, col), direction)
                
                if can_build:
                    dr, dc = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}[direction]
                    to_pos = (row + dr, col + dc)
                    
                    self.pathfinder.addConnection((row, col), to_pos)
                    edges_added += 1
                    
                    # Для разноуровневых секторов (пандус) - одностороннее ребро
                    if sector.floor_level != self.sectors[to_pos].floor_level:
                        # Пандус односторонний
                        self.pathfinder.addConnection((row, col), to_pos, oneWay=True)
                        edges_added += 1
        
        logging.info(f"Построено ребер: {edges_added}")
    
    def get_pipe_positions(self, color: str) -> List[Pipe]:
        """
        Получить позиции труб определенного цвета.
        
        Args:
            color: Цвет трубы ('green', 'blue', 'red')
            
        Returns:
            List[Pipe]: Список труб указанного цвета
        """
        if color == 'green':
            return self.green_pipes
        elif color == 'blue':
            return self.blue_pipes
        elif color == 'red':
            return self.red_pipes
        return []
    
    def get_all_pipes(self) -> List[Pipe]:
        """Получить все трубы."""
        return self.green_pipes + self.blue_pipes + self.red_pipes
    
    def get_unvisited_sectors(self) -> List[Tuple[int, int]]:
        """
        Получить список непосещенных секторов.
        
        Returns:
            List[Tuple[int, int]]: Список координат непосещенных секторов
        """
        return [(r, c) for (r, c), s in self.sectors.items() 
                if not s.visited and s.floor_level != FloorLevel.UNKNOWN]
    
    def get_explored_sectors(self) -> List[Tuple[int, int]]:
        """Получить список исследованных секторов."""
        return [(r, c) for (r, c), s in self.sectors.items() if s.visited]
    
    def is_field_explored(self, required_green: int = 3, required_colored: int = 3) -> bool:
        """
        Проверить, исследованно ли все поле.
        
        Условия:
        - Найдены 3+ зеленые трубы
        - Найдены суммарно 3+ синие и красные трубы
        
        Args:
            required_green: Требуемое количество зеленых труб
            required_colored: Требуемое количество цветных труб
            
        Returns:
            bool: True если поле исследованно
        """
        green_count = len(self.green_pipes)
        colored_count = len(self.blue_pipes) + len(self.red_pipes)
        
        return green_count >= required_green and colored_count >= required_colored
    
    def get_nearest_unvisited(self, start: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Найти ближайший непосещенный сектор.
        
        Args:
            start: Начальная позиция (row, col)
            
        Returns:
            Tuple[int, int] или None: Координаты ближайшего непосещенного сектора
        """
        # Сначала ищем среди непосещенных с известным уровнем
        unvisited = self.get_unvisited_sectors()
        
        if not unvisited:
            return None
        
        # Если граф построен, используем маршрутизацию
        try:
            best_target = None
            best_distance = float('inf')
            best_path = None
            
            for target in unvisited:
                distance, commands, path = self.pathfinder.getRoute(
                    start, target, start_direction=self.current_direction
                )
                if distance > 0 and distance < best_distance:
                    best_distance = distance
                    best_target = target
                    best_path = path
            
            return best_target
            
        except (ValueError, IndexError):
            # Если маршрутизация не работает, ищем ближайший по Манхэттену
            def manhattan(pos1, pos2):
                return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
            
            return min(unvisited, key=lambda p: manhattan(start, p))
    
    def find_path_to(self, start: Tuple[int, int], end: Tuple[int, int]) -> Tuple[int, str, List[Tuple]]:
        """
        Найти путь от start до end.
        
        Args:
            start: Начальная позиция (row, col)
            end: Конечная позиция (row, col)
            
        Returns:
            Tuple[int, str, List[Tuple]]: (расстояние, команды, путь)
        """
        return self.pathfinder.getRoute(start, end, start_direction=self.current_direction)
    
    def get_sector_center(self, row: int, col: int) -> Tuple[float, float]:
        """
        Получить координаты центра сектора в пикселях.
        
        Args:
            row: Строка
            col: Столбец
            
        Returns:
            Tuple[float, float]: (x, y) координаты центра
        """
        # Предполагаем что сетка выровнена по центрам перекрестков
        # Эти значения должны быть синхронизированы с testing.py
        GRID_SPACING_X = 100  # пикселей между центрами по X
        GRID_SPACING_Y = 100  # пикселей между центрами по Y
        
        return (col * GRID_SPACING_X + GRID_SPACING_X // 2,
                row * GRID_SPACING_Y + GRID_SPACING_Y // 2)
    
    def set_current_position(self, row: int, col: int, direction: Optional[str] = None):
        """
        Установить текущую позицию робота.
        
        Args:
            row: Строка
            col: Столбец
            direction: Направление (опционально)
        """
        self.current_position = (row, col)
        if direction:
            self.current_direction = direction
        logging.info(f"Позиция: ({row}, {col}), напр: {self.current_direction}")
    
    def visualize(self, path: Optional[List[Tuple[int, int]]] = None):
        """
        Визуализировать граф.
        
        Args:
            path: Опциональный путь для отображения
        """
        self.pathfinder.visualize(path)
    
    def get_status(self) -> dict:
        """
        Получить статус графа.
        
        Returns:
            dict: Статистика графа
        """
        visited = sum(1 for s in self.sectors.values() if s.visited)
        return {
            'grid_size': f"{self.grid_rows}x{self.grid_cols}",
            'total_sectors': len(self.sectors),
            'visited_sectors': visited,
            'exploration_percent': 100 * visited / len(self.sectors),
            'green_pipes': len(self.green_pipes),
            'blue_pipes': len(self.blue_pipes),
            'red_pipes': len(self.red_pipes),
            'ramps': len(self.ramps),
            'is_explored': self.is_field_explored()
        }


def analyze_sector_image(image_path: str) -> dict:
    """
    Анализировать изображение сектора.
    
    Wrapper для использования analyse.analyze_image
    
    Args:
        image_path: Путь к изображению
        
    Returns:
        dict: Результат анализа
    """
    from analyse import analyze_image
    return analyze_image(image_path)


# =============================================================================
# ТЕСТИРОВАНИЕ
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Создаем тестовый граф
    graph = BuildGraph(grid_rows=4, grid_cols=4)
    
    # Симулируем анализ секторов
    test_analyses = [
        {'black_area_px': 5000, 'has_green': True, 'blue_area_px': 0, 'red_area_px': 0},  # (0,0) белый
        {'black_area_px': 15000, 'has_green': False, 'blue_area_px': 0, 'red_area_px': 0},  # (0,1) черный
        {'black_area_px': 5000, 'has_green': False, 'blue_area_px': 0, 'red_area_px': 0},   # (0,2) белый
        {'black_area_px': 5000, 'has_green': False, 'blue_area_px': 500, 'red_area_px': 0}, # (0,3) белый + синяя
        {'black_area_px': 15000, 'has_green': False, 'blue_area_px': 0, 'red_area_px': 400},# (1,0) черный + красная
    ]
    
    positions = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)]
    
    for pos, analysis in zip(positions, test_analyses):
        graph.add_sector_analysis(pos[0], pos[1], analysis)
    
    # Строим ребра
    graph.build_edges()
    
    # Проверяем статус
    print("\n=== Статус графа ===")
    status = graph.get_status()
    for key, value in status.items():
        print(f"{key}: {value}")
    
    print(f"\nЗеленые трубы: {[p.sector for p in graph.green_pipes]}")
    print(f"Синие трубы: {[p.sector for p in graph.blue_pipes]}")
    print(f"Красные трубы: {[p.sector for p in graph.red_pipes]}")
    
    # Ищем ближайший непосещенный
    nearest = graph.get_nearest_unvisited((0, 0))
    print(f"\nБлижайший непосещенный от (0,0): {nearest}")
    
    # Находим путь
    if nearest:
        dist, commands, path = graph.find_path_to((0, 0), nearest)
        print(f"Путь до ({nearest[0]},{nearest[1]}): dist={dist}, commands={commands}")