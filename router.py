'''
Реализован класс PathFinder который создает граф NxN вершин и дает возможность создавать ребра(одно и двунаправленные)
Выдает Расстояние, строку команд и список координат пути
visualize работает как для графа, так и для пути
'''
import heapq
from typing import Tuple, List, Optional
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

class Pathfinder:
    def __init__(self, n: int):
        self.n = n
        self.graph = [[set() for _ in range(n)] for _ in range(n)]
        
        # Базовые направления: Верх, Право, Низ, Лево
        self.dirs = [(-1, 0, 'U'), (0, 1, 'R'), (1, 0, 'D'), (0, -1, 'L')]
        self.dir_map = {'U': 0, 'R': 1, 'D': 2, 'L': 3}
        self.turn_map = {1: 'R', 2: 'A', 3: 'L'}
        
    def addConnection(self, p1: Tuple[int, int], p2: Tuple[int, int], oneWay: bool = False):
        """
        Добавляет ребро между двумя точками.
        По умолчанию ребро двунаправленное. Если oneWay=True, то направление только от p1 к p2.
        """
        r1, c1 = p1
        r2, c2 = p2
        if not (0 <= r1 < self.n and 0 <= c1 < self.n and 0 <= r2 < self.n and 0 <= c2 < self.n):
            raise ValueError("Точки находятся за пределами сетки")
            
        self.graph[r1][c1].add(p2)
        if not oneWay:
            self.graph[r2][c2].add(p1)
            
    def getRoute(self, start: Tuple[int, int], end: Tuple[int, int], start_direction: str = 'U') -> Tuple[int, str, List[Tuple[int, int]]]:
        """
        Ищет путь и возвращает длину, строку команд (F, R, L, A) и список координат пути.
        start_direction: начальное направление робота ('U', 'R', 'D', 'L').
        """
        if start_direction not in self.dir_map:
            raise ValueError("start_direction must be 'U', 'R', 'D', or 'L'")

        sr, sc = start
        er, ec = end
        if not (0 <= sr < self.n and 0 <= sc < self.n and 0 <= er < self.n and 0 <= ec < self.n):
            raise ValueError("Точки находятся за пределами сетки")
            
        if start == end:
            return 0, "", [start]
            
        pq = []
        INF = float('inf')
        
        dist = [[[INF] * 4 for _ in range(self.n)] for _ in range(self.n)]
        turns = [[[INF] * 4 for _ in range(self.n)] for _ in range(self.n)]
        parent = [[[None] * 4 for _ in range(self.n)] for _ in range(self.n)]
        
        for d_idx, (dr, dc, cmd) in enumerate(self.dirs):
            nr, nc = sr + dr, sc + dc
            if 0 <= nr < self.n and 0 <= nc < self.n and (nr, nc) in self.graph[sr][sc]:
                dist[nr][nc][d_idx] = 1
                turns[nr][nc][d_idx] = 0
                parent[nr][nc][d_idx] = (sr, sc, -1, cmd)
                heapq.heappush(pq, (1, 0, nr, nc, d_idx))
                
        while pq:
            d, t, r, c, d_idx = heapq.heappop(pq)
            
            if (r, c) == end:
                # 1. Восстановление базового пути
                path_cmds = []
                path_coords = []
                curr_r, curr_c, curr_d = r, c, d_idx
                while curr_d != -1:
                    pr, pc, pd_idx, cmd = parent[curr_r][curr_c][curr_d]
                    path_cmds.append(cmd)
                    path_coords.append((curr_r, curr_c))
                    curr_r, curr_c, curr_d = pr, pc, pd_idx
                
                path_coords.append((sr, sc))
                path_cmds.reverse()
                path_coords.reverse()
                
                # 2. Форматирование в команды F, R, L, A
                curr_dir = self.dir_map[start_direction]
                commands = []
                f_count = 0
                
                for cmd in path_cmds:
                    next_dir_idx = self.dir_map[cmd]
                    diff = (next_dir_idx - curr_dir) % 4
                    
                    if diff != 0:
                        if f_count > 0:
                            commands.append(f"F{f_count}")
                            f_count = 0
                        commands.append(self.turn_map[diff])
                        curr_dir = next_dir_idx
                        
                    f_count += 1
                    
                if f_count > 0:
                    commands.append(f"F{f_count}")
                    
                return d, "".join(commands), path_coords
                
            if d > dist[r][c][d_idx] or (d == dist[r][c][d_idx] and t > turns[r][c][d_idx]):
                continue
                
            for nr, nc in self.graph[r][c]:
                dr = nr - r
                dc = nc - c
                next_d_idx = -1
                for idx, (ddr, ddc, _) in enumerate(self.dirs):
                    if ddr == dr and ddc == dc:
                        next_d_idx = idx
                        break
                        
                if next_d_idx == -1:
                    continue
                    
                cmd = self.dirs[next_d_idx][2]
                new_d = d + 1
                new_t = t + (1 if next_d_idx != d_idx else 0)
                
                if new_d < dist[nr][nc][next_d_idx] or \
                   (new_d == dist[nr][nc][next_d_idx] and new_t < turns[nr][nc][next_d_idx]):
                    
                    dist[nr][nc][next_d_idx] = new_d
                    turns[nr][nc][next_d_idx] = new_t
                    parent[nr][nc][next_d_idx] = (r, c, d_idx, cmd)
                    heapq.heappush(pq, (new_d, new_t, nr, nc, next_d_idx))
                    
        return -1, "", []

    def visualize(self, path: Optional[List[Tuple[int, int]]] = None):
        """
        Визуализация графа с автоматическим масштабированием под размер окна (макс 1400x1400).
        """
        if not HAS_CV2:
            print("Для визуализации требуется библиотека opencv-python. Установите её командой: pip install opencv-python")
            return

        MAX_IMG_SIZE = 700

        padding = 60
        available_size = MAX_IMG_SIZE - padding
        
        # Автоматический расчет размера ячейки
        cell_size = available_size // self.n
        cell_size = max(20, cell_size) # Минимальный размер ячейки для читаемости
        
        img_size = self.n * cell_size + padding
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
        
        # Адаптивные размеры элементов
        node_radius = max(4, cell_size // 5)
        arrow_len = max(6, cell_size // 6)
        thickness = max(1, cell_size // 50)
        path_thickness = max(2, thickness * 2)
        font_scale = min(0.5, cell_size / 100.0)
        
        def get_center(r, c):
            return (c * cell_size + cell_size // 2 + padding // 2, 
                    r * cell_size + cell_size // 2 + padding // 2)
            
        def draw_arrow(x1, y1, x2, y2, color, t=thickness):
            dx = x2 - x1
            dy = y2 - y1
            length = np.hypot(dx, dy)
            
            if length > 0:
                ux, uy = dx / length, dy / length
                x2 = int(x2 - ux * node_radius)
                y2 = int(y2 - uy * node_radius)
            
            cv2.line(img, (x1, y1), (x2, y2), color, t)
            
            angle = np.arctan2(y2 - y1, x2 - x1)
            arrow_angle = np.pi / 6
            ax1 = int(x2 - arrow_len * np.cos(angle - arrow_angle))
            ay1 = int(y2 - arrow_len * np.sin(angle - arrow_angle))
            ax2 = int(x2 - arrow_len * np.cos(angle + arrow_angle))
            ay2 = int(y2 - arrow_len * np.sin(angle + arrow_angle))
            cv2.fillPoly(img, [np.array([[x2, y2], [ax1, ay1], [ax2, ay2]])], color)

        # Отрисовка ребер графа
        drawn = set()
        total_edges = 0
        one_way_edges = 0
        two_way_edges = 0
        
        for r in range(self.n):
            for c in range(self.n):
                for nr, nc in self.graph[r][c]:
                    edge_pair = tuple(sorted(((r, c), (nr, nc))))
                    if edge_pair in drawn:
                        continue
                    drawn.add(edge_pair)
                    
                    forward = (nr, nc) in self.graph[r][c]
                    backward = (r, c) in self.graph[nr][nc]
                    
                    x1, y1 = get_center(r, c)
                    x2, y2 = get_center(nr, nc)
                    
                    if forward and backward:
                        cv2.line(img, (x1, y1), (x2, y2), (150, 150, 150), thickness)
                        two_way_edges += 1
                    elif forward and not backward:
                        draw_arrow(x1, y1, x2, y2, (100, 100, 200))
                        one_way_edges += 1
                    elif backward and not forward:
                        draw_arrow(x2, y2, x1, y1, (100, 100, 200))
                        one_way_edges += 1

        # Отрисовка пути поверх графа
        if path and len(path) > 1:
            for i in range(len(path) - 1):
                r1, c1 = path[i]
                r2, c2 = path[i+1]
                x1, y1 = get_center(r1, c1)
                x2, y2 = get_center(r2, c2)
                draw_arrow(x1, y1, x2, y2, (0, 200, 0), t=path_thickness)

        # Отрисовка узлов
        for r in range(self.n):
            for c in range(self.n):
                x, y = get_center(r, c)
                if path and (r, c) == path[0]:
                    color = (0, 165, 255)  # Оранжевый для старта
                elif path and (r, c) == path[-1]:
                    color = (0, 0, 255)    # Красный для финиша
                else:
                    color = (50, 50, 50)   # Темно-серый для обычных узлов
                
                cv2.circle(img, (x, y), node_radius, color, -1)
                
                # Подписи координат, только если размер ячейки позволяет
                if cell_size >= 40:
                    coord_text = f"({r},{c})"
                    text_size = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
                    cv2.putText(img, coord_text, (x - text_size[0]//2, y + node_radius + text_size[1] + 2), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (100, 100, 100), 1)

        print(f"\n=== Статистика графа ===")
        print(f"Размер сетки: {self.n}x{self.n}")
        print(f"Всего узлов: {self.n * self.n}")
        print(f"Двусторонних ребер: {two_way_edges}")
        print(f"Односторонних ребер: {one_way_edges}")
        print(f"Всего ребер: {two_way_edges + one_way_edges}")
        if path:
            print(f"Путь найден и отображен зеленым цветом")
        print("========================\n")
                
        cv2.imshow("Graph Visualization", img)
        print("Нажмите любую клавишу в окне визуализации для закрытия...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Пример использования
if __name__ == "__main__":
    # Создаем большую сетку 16x16
    pf = Pathfinder(16)

    # Добавляем связи (строим граф)
    # Горизонтальные связи
    for r in range(16):
        for c in range(15):
            pf.addConnection((r, c), (r, c+1))
            
    # Вертикальные связи
    for c in range(16):
        for r in range(15):
            pf.addConnection((r, c), (r+1, c))
            
    # Добавим одностороннее ребро
    pf.addConnection((0, 0), (1, 1), oneWay=True)

    # Ищем путь
    length, commands, path = pf.getRoute((0,0), (15,15))
    print(f"Длина пути: {length}, Команды: {commands[:50]}...") # Выводим только начало строки
    
    # Визуализируем граф и путь (автоматически масштабируется под 1400x1400)
    pf.visualize()