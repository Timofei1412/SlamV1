import heapq
from typing import Tuple

class Pathfinder:
    def __init__(self, n: int):
        self.n = n
        self.graph = [[set() for _ in range(n)] for _ in range(n)]
        
        # Базовые направления: Верх, Право, Низ, Лево
        self.dirs = [(-1, 0, 'U'), (0, 1, 'R'), (1, 0, 'D'), (0, -1, 'L')]
        self.dir_map = {'U': 0, 'R': 1, 'D': 2, 'L': 3}
        self.turn_map = {1: 'R', 2: 'A', 3: 'L'}
        
    def addConnection(self, p1: Tuple[int, int], p2: Tuple[int, int]):
        r1, c1 = p1
        r2, c2 = p2
        if not (0 <= r1 < self.n and 0 <= c1 < self.n and 0 <= r2 < self.n and 0 <= c2 < self.n):
            raise ValueError("Точки находятся за пределами сетки")
            
        self.graph[r1][c1].add(p2)
        self.graph[r2][c2].add(p1)
        
    def getRoute(self, start: Tuple[int, int], end: Tuple[int, int], start_direction: str = 'U') -> Tuple[int, str]:
        """
        Ищет путь и возвращает длину и строку команд (F, R, L, A).
        start_direction: начальное направление робота ('U', 'R', 'D', 'L').
        """
        if start_direction not in self.dir_map:
            raise ValueError("start_direction must be 'U', 'R', 'D', or 'L'")

        sr, sc = start
        er, ec = end
        if not (0 <= sr < self.n and 0 <= sc < self.n and 0 <= er < self.n and 0 <= ec < self.n):
            raise ValueError("Точки находятся за пределами сетки")
            
        if start == end:
            return 0, ""
            
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
                # 1. Восстановление базового пути (список направлений)
                path_cmds = []
                curr_r, curr_c, curr_d = r, c, d_idx
                while curr_d != -1:
                    pr, pc, pd_idx, cmd = parent[curr_r][curr_c][curr_d]
                    path_cmds.append(cmd)
                    curr_r, curr_c, curr_d = pr, pc, pd_idx
                path_cmds.reverse()
                
                # 2. Форматирование в команды F, R, L, A
                curr_dir = self.dir_map[start_direction]
                commands = []
                f_count = 0
                
                for cmd in path_cmds:
                    next_dir_idx = self.dir_map[cmd]
                    diff = (next_dir_idx - curr_dir) % 4
                    
                    # Если направление изменилось
                    if diff != 0:
                        if f_count > 0:
                            commands.append(f"F{f_count}")
                            f_count = 0
                        commands.append(self.turn_map[diff])
                        curr_dir = next_dir_idx
                        
                    f_count += 1
                    
                if f_count > 0:
                    commands.append(f"F{f_count}")
                    
                return d, "".join(commands)
                
            # Пропуск неоптимальных состояний
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
                    
        return -1, ""

# Создаем сетку 3x3
pf = Pathfinder(3)

# Добавляем связи (строим лабиринт)
# (0,0) -> (0,1) -> (0,2)
pf.addConnection((0,0), (0,1))
pf.addConnection((0,1), (0,2))
# (0,1) -> (1,1) -> (2,1) -> (2,2)
pf.addConnection((0,1), (1,1))
pf.addConnection((1,1), (2,1))
pf.addConnection((2,1), (2,2))

# Ищем путь из (0,0) в (2,2)
length, commands = pf.getRoute((0,0), (2,2))
print(f"Длина: {length}, Путь: {commands}") 
# Ожидаемый вывод: Длина: 4, Путь: RRDD или DDRR (в зависимости от приоритета направлений, но с минимумом поворотов)