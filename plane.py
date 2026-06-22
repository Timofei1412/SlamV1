import cv2
import numpy as np
import pickle

def resize_to_max(img, max_size=1000):
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img, 1.0
    scale = max_size / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale))), scale

# === Загрузка карт ===
print("Загрузка карт...")
with open('mirror_maps.pkl', 'rb') as f:
    data = pickle.load(f)

map_x = data['map_x']
map_y = data['map_y']
valid_mask = data['valid_mask']
params = data['params']

print(f"Размер карт: {map_x.shape}")
print(f"Диапазон map_x: [{map_x[valid_mask].min():.0f}, {map_x[valid_mask].max():.0f}]")
print(f"Диапазон map_y: [{map_y[valid_mask].min():.0f}, {map_y[valid_mask].max():.0f}]")
print(f"Размер исходного изображения (из карт): {params['img_size']}")

# === Загрузка ИСХОДНОГО изображения (БЕЗ уменьшения!) ===
img = cv2.imread('Images/photo3.jpg')
if img is None:
    print("ОШИБКА: не удалось загрузить mirror.jpg")
    exit()

h_img, w_img = img.shape[:2]
print(f"Исходное изображение: {w_img}x{h_img}")

# Проверка соответствия размеров
expected_w, expected_h = params['img_size']
if w_img != expected_w or h_img != expected_h:
    print(f"⚠️  ВНИМАНИЕ: размер изображения не совпадает с картами!")
    print(f"   Ожидалось: {expected_w}x{expected_h}, получено: {w_img}x{h_img}")
    print(f"   Пересчитайте карты для текущего изображения.")

# === Применяем remap к ПОЛНОМУ изображению ===
print("Применение преобразования...")
result = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderValue=(0, 0, 0))

# === Диагностика ===
non_zero = np.sum(np.any(result > 0, axis=2))
total = result.shape[0] * result.shape[1]
print(f"Непустых пикселей: {non_zero} ({100*non_zero/total:.1f}%)")

# === Уменьшаем ТОЛЬКО ДЛЯ ОТОБРАЖЕНИЯ ===
result_display, _ = resize_to_max(result, 1000)
img_display, _ = resize_to_max(img, 1000)

# === Показ ===
cv2.imshow('Original', img_display)
cv2.imshow('Result', result_display)
print("Нажмите любую клавишу...")
cv2.waitKey(0)
cv2.destroyAllWindows()

# === Сохраняем полное изображение ===
# cv2.imwrite('result_full.png', result)
# print("Сохранено: result_full.png")