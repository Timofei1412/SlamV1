import cv2
import numpy as np
import math
import pickle


# === Параметры системы (все в мм и градусах) ===
H_tip = 237             # Высота вершины конуса от пола (мм)
alpha_deg = 20           # Полуугол конуса (градусы)
D = 40                   # Диаметр основания конуса (мм)
R_base = D / 2           # Радиус основания (мм)
distance_camera_to_tip = 10 # Расстояние от камеры до вершины конуса (мм)


# Параметры камеры
FOV_deg = 25             # Угол обзора камеры (градусы)


# Параметры выходного изображения
out_res = 1000           # Размер выходного изображения (пиксели)


# === Вычисляем высоту камеры ===
camera_height = H_tip - distance_camera_to_tip
print(f"Высота камеры от пола: {camera_height:.2f} мм")
print(f"Расстояние от камеры до вершины конуса: {distance_camera_to_tip:.2f} мм")


# === Класс PointSelector (Интерактивный выбор параметров) ===
class PointSelector:
    def __init__(self, img):
        self.img = img.copy()
        self.display = img.copy()
        self.center = None
        self.inner_points = []
        self.outer_points = []
        self.mode = 'center'
        self.window_name = 'Select cone: [C]enter, [I]nner, [O]uter, [ENTER/Q] - done'
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.update_display()
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == 'center':
                self.center = (x, y)
                self.mode = 'inner'
                print(f"Центр: ({x}, {y}). Теперь укажите точки вершины конуса")
            elif self.mode == 'inner':
                self.inner_points.append((x, y))
                if len(self.inner_points) < 15: # Ограничение для стабильности
                    print(f"Точка вершины добавлена. Всего точек: {len(self.inner_points)}")
            elif self.mode == 'outer':
                self.outer_points.append((x, y))
                if len(self.outer_points) < 15: # Ограничение для стабильности
                    print(f"Точка основания добавлена. Всего точек: {len(self.outer_points)}")
            self.update_display()
    
    def update_display(self):
        self.display = self.img.copy()
        
        if self.center is not None:
            cv2.circle(self.display, self.center, 5, (0, 255, 0), -1)
            
            if len(self.inner_points) > 0:
                r_inner = np.mean([np.sqrt((p[0]-self.center[0])**2 + (p[1]-self.center[1])**2) 
                                     for p in self.inner_points])
                cv2.circle(self.display, self.center, int(r_inner), (0, 255, 255), 2)
                for p in self.inner_points:
                    cv2.circle(self.display, p, 3, (0, 255, 255), -1)
            
            if len(self.outer_points) > 0:
                r_outer = np.mean([np.sqrt((p[0]-self.center[0])**2 + (p[1]-self.center[1])**2) 
                                     for p in self.outer_points])
                cv2.circle(self.display, self.center, int(r_outer), (255, 0, 0), 2)
                for p in self.outer_points:
                    cv2.circle(self.display, p, 3, (255, 0, 0), -1)
        
        mode_text = f"Mode: {self.mode.upper()}"
        cv2.putText(self.display, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow(self.window_name, self.display)
    
    def run(self):
        print("\n=== Интерактивный выбор параметров конуса ===")
        print("1. Кликните на ЦЕНТР конуса (Нажмите 'C')")
        print("2. Нажмите 'I' и кликайте по ВЕРШИНЕ конуса (для внутренних точек)")
        print("3. Нажмите 'O' и кликайте по ОСНОВАНИЮ конуса (для внешних точек)")
        print("4. Нажмите ENTER или Q, когда закончите")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c') or key == ord('C') or key == ord('с') or key == ord('С'):
                self.mode = 'center'
            elif key == ord('i') or key == ord('I') or key == ord('ш') or key == ord('Ш'):
                self.mode = 'inner'
            elif key == ord('o') or key == ord('O') or key == ord('щ') or key == ord('Щ'):
                self.mode = 'outer'
            elif key == 13 or key == ord('q') or key == ord('Q'):
                if self.center is not None and len(self.inner_points) > 0 and len(self.outer_points) > 0:
                    break
                else:
                    print("Требуется задать Центр, внутренние И и внешние О точки.")
            
            cv2.imshow(self.window_name, self.display)

        cv2.destroyAllWindows()
        
        if not self.inner_points:
             r_inner = 0
        else:
             r_inner = np.mean([np.sqrt((p[0]-self.center[0])**2 + (p[1]-self.center[1])**2) for p in self.inner_points])

        if not self.outer_points:
            r_outer = 0
        else:
            r_outer = np.mean([np.sqrt((p[0]-self.center[0])**2 + (p[1]-self.center[1])**2) for p in self.outer_points])

        print(f"\nРезультаты:")
        print(f"Центр: {self.center}")
        print(f"Внутренний радиус: {r_inner:.1f} пикселей")
        print(f"Внешний радиус: {r_outer:.1f} пикселей")
        
        return self.center, int(r_inner), int(r_outer)


# === Загрузка изображения ===
img_path = 'Images/photo3.jpg'
try:
    img = cv2.imread(img_path)
except Exception as e:
    print(f"Ошибка при чтении файла: {e}")
    exit()

if img is None:
    print(f"Ошибка: не удалось загрузить {img_path}. Проверьте путь.")
    exit()


h_img, w_img = img.shape[:2]
print(f"\nИсходное изображение: {w_img}x{h_img}")


# Уменьшаем для отображения (если исходник слишком большой)
display_max_size = 1000
scale_factor = display_max_size / max(h_img, w_img)

if scale_factor < 1.0:
    display_w = int(w_img * scale_factor)
    display_h = int(h_img * scale_factor)
    img_display = cv2.resize(img, (display_w, display_h))
else:
    img_display = img.copy()


# Интерактивный выбор
selector = PointSelector(img_display)
center_disp, r_min_disp, r_max_disp = selector.run()


# Масштабирование координат с дисплейного изображения обратно к исходному размеру (w_img, h_img)
cx = int(center_disp[0] / scale_factor)
cy = int(center_disp[1] / scale_factor)
r_min = int(r_min_disp / scale_factor)
r_max = int(r_max_disp / scale_factor)


print(f"\nПараметры для исходного изображения:")
print(f"Центр: ({cx}, {cy})")
print(f"Внутренний радиус (пикс): {r_min}")
print(f"Внешний радиус (пикс): {r_max}")


# === Расчёт фокусного расстояния (фокальное расстояние) ===
alpha = math.radians(alpha_deg)
FOV = math.radians(FOV_deg)
f = w_img / (2 * math.tan(FOV / 2))
print(f"\nФокусное расстояние камеры (F): {f:.1f} пикселей")


# === Расчёт 3D координат с учётом высоты камеры и перспективы ===
print("\nРасчёт 3D координат на конусе...")

y_grid, x_grid = np.mgrid[0:h_img, 0:w_img].astype(np.float64)
dx = (x_grid - cx).astype(np.float64)
dy = (y_grid - cy).astype(np.float64)


# Итеративный расчёт Z (Высота точки на конусе)
z = np.full_like(dx, H_tip, dtype=np.float64)

print("Начало итеративного расчёта высот...")
max_iter = 30 # Увеличиваем лимит для стабильности
for iteration in range(max_iter):
    # Расстояние от камеры до точки по Z (Z - CameraHeight)
    z_relative = z - camera_height
    
    x_3d = dx * z_relative / f
    y_3d = dy * z_relative / f
    r_3d = np.sqrt(x_3d**2 + y_3d**2)
    
    z_new = H_tip + r_3d / math.tan(alpha)
    
    max_diff = np.max(np.abs(z_new - z))
    if max_diff < 0.1: # Критерий остановки немного смягчен для надежности
        print(f"Сходимость достигнута на итерации {iteration+1} (макс. разница={max_diff:.4f})")
        z = z_new
        break
    
    z = z_new
else:
    print("Предупреждение: Достигнут максимальный лимит итераций.")


# Финальные 3D координаты (X, Y, Z)
x_3d = dx * (z - camera_height) / f
y_3d = dy * (z - camera_height) / f
r_3d = np.sqrt(x_3d**2 + y_3d**2)


print(f"Диапазон высот на конусе: [{z.min():.2f}, {z.max():.2f}] мм")
print(f"Диапазон радиусов: [{r_3d.min():.2f}, {r_3d.max():.2f}] мм")


# Маска конуса (Только точки, которые находятся на поверхности и внутри основания)
mask = (r_3d >= 0) & (r_3d <= R_base)
print(f"Пикселей в маске конуса: {np.sum(mask)}")


# === Лучи от камеры к точкам на конусе (Ray Tracing) ===

# Камера находится в точке (0, 0, camera_height) относительно себя.
ray_x = x_3d
ray_y = y_3d
ray_z = z - camera_height  # Относительный Z: от камеры до точки на конусе


# Нормали к внутренней поверхности конуса
theta = np.arctan2(y_3d, x_3d)
n_x = np.cos(theta) * math.sin(alpha) 
n_y = np.sin(theta) * math.sin(alpha)
n_z = -math.cos(alpha)


# Нормализация вектора нормали N (Обеспечиваем, что нет деления на ноль!)
n_norm = np.sqrt(n_x**2 + n_y**2 + n_z**2)
# Используем np.divide для безопасного поэлементного деления: 
n_x = np.where(n_norm != 0, n_x / n_norm, n_x)
n_y = np.where(n_norm != 0, n_y / n_norm, n_y)
n_z = np.where(n_norm != 0, n_z / n_norm, n_z)


# --- ИСПРАВЛЕНИЕ ЗДЕСЬ: Безопасная нормализация луча ---
ray_norm = np.sqrt(ray_x**2 + ray_y**2 + ray_z**2)

# Создаем маску для всех точек, где норма не равна нулю
safe_mask = (ray_norm != 0)

# Инициализируем новые векторы нулями
rx_new = np.zeros_like(ray_x)
ry_new = np.zeros_like(ray_y)
rz_new = np.zeros_like(ray_z)

# Выполняем нормализацию только там, где это безопасно
rx_new[safe_mask] = ray_x[safe_mask] / ray_norm[safe_mask]
ry_new[safe_mask] = ray_y[safe_mask] / ray_norm[safe_mask]
rz_new[safe_mask] = ray_z[safe_mask] / ray_norm[safe_mask]

# Обновляем переменные для дальнейших расчетов
ray_x, ray_y, ray_z = rx_new, ry_new, rz_new
# ------------------------------------------------------


# Отражение
dot = ray_x * n_x + ray_y * n_y + ray_z * n_z
ref_x = ray_x - 2 * dot * n_x
ref_y = ray_y - 2 * dot * n_y
ref_z = ray_z - 2 * dot * n_z


# === Проекция на пол (Z=0) ===

valid_reflection = (ref_z < 0) & mask

# Вычисляем параметр времени t: z + t * ref_z = 0 => t = -z / ref_z
t_ref = np.where(valid_reflection, -z / ref_z, 0)

# Координаты на полу (Z=0): P_x = x_3d + t * ref_x; P_y = y_3d + t * ref_y
floor_x = np.where(valid_reflection, x_3d + t_ref * ref_x, 0)
floor_y = np.where(valid_reflection, y_3d + t_ref * ref_y, 0)


# Собираем валидные точки для маппинга
valid_indices = np.where(valid_reflection)
valid_floor_x = floor_x[valid_indices]
valid_floor_y = floor_y[valid_indices]
valid_src_x = x_grid[valid_indices].astype(np.float32)
valid_src_y = y_grid[valid_indices].astype(np.float32)


print(f"\nВалидных точек: {len(valid_floor_x)}")


# === КОРРЕКТИРОВАННАЯ СЕКЦИЯ: ОПРЕДЕЛЕНИЕ БОРДЮРА И МАСШТАБИРОВАНИЕ ===

if len(valid_floor_x) == 0:
    print("ОШИБКА: Нет валидных точек для проекции.")
    exit()

# 1. Определяем физический диапазон (в мм)
min_x, max_x = valid_floor_x.min(), valid_floor_x.max()
min_y, max_y = valid_floor_y.min(), valid_floor_y.max()

# 2. Определяем желаемый физический запас (margin). Используем фиксированный процент от диапазона.
margin_factor = 0.15 # Увеличиваем запас до 15% для лучшего покрытия краев
range_x_span = max(max_x - min_x, 1e-6)
range_y_span = max(max_y - min_y, 1e-6)

margin_x = margin_factor * range_x_span
margin_y = margin_factor * range_y_span

# Обновленные границы: [min_x', max_x'] и [min_y', max_y']
final_min_x = min_x - margin_x
final_max_x = max_x + margin_x
final_min_y = min_y - margin_y
final_max_y = max_y + margin_y

range_x = final_max_x - final_min_x
range_y = final_max_y - final_min_y


# 3. Расчет масштабирования (Масштаб должен сопоставить крайнюю точку [0] и последнюю точку [N-1])
scale_x = (out_res - 1) / range_x if range_x != 0 else 0
scale_y = (out_res - 1) / range_y if range_y != 0 else 0


# Создаем карты
map_x = np.zeros((out_res, out_res), dtype=np.float32)
map_y = np.zeros((out_res, out_res), dtype=np.float32)
valid_mask = np.zeros((out_res, out_res), dtype=bool)


print("\nЗаполнение карт с использованием корректного масштабирования...")
for i in range(len(valid_floor_x)):
    # 1. Сдвигаем координату от минимальной границы (final_min)
    shifted_x = valid_floor_x[i] - final_min_x
    shifted_y = valid_floor_y[i] - final_min_y
    
    # 2. Масштабируем: P_pixel = floor(shifted * scale)
    px_out = int(shifted_x * scale_x)
    py_out = int(shifted_y * scale_y)
    
    # Проверка границ
    if 0 <= px_out < out_res and 0 <= py_out < out_res:
        map_x[py_out, px_out] = valid_floor_x[i] # Сохраняем исходное значение X в map_x
        map_y[py_out, px_out] = valid_floor_y[i] # Сохраняем исходное значение Y в map_y
        valid_mask[py_out, px_out] = True


print(f"Заполнено пикселей: {np.sum(valid_mask)}")

# === СОХРАНЕНИЕ РЕЗУЛЬТАТА ===
data = {
    'map_x': map_x,
    'map_y': map_y,
    'valid_mask': valid_mask,
    'params': {
        'min_x': final_min_x, 'max_x': final_max_x, # Сохраняем финальные границы
        'min_y': final_min_y, 'max_y': final_max_y,
        'out_res': out_res,
        'cx': cx, 'cy': cy,
        'r_min': r_min, 'r_max': r_max,
        'img_size': (w_img, h_img),
        'H_tip': H_tip,
        'alpha_deg': alpha_deg,
        'R_base': R_base,
        'f': f,
        'camera_height': camera_height,
        'distance_camera_to_tip': distance_camera_to_tip
    }
}


with open('mirror_maps.pkl', 'wb') as f:
    pickle.dump(data, f)


print(f"\n[УСПЕХ] Карты сохранены в mirror_maps.pkl")
