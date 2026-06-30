'''
Анализирует изображения 1.png, 2.png, 3.png, 4.png из папки Output и сохраняет данные в .json файл
'''
import cv2
import numpy as np
import os
import json
import glob

# --- Пороговые значения для настройки (в пикселях) ---
BLACK_AREA_FLOOR_THRESHOLD = 10000
BLACK_AREA_MIN_THRESHOLD = 10000
GREEN_AREA_THRESHOLD = 500

def analyze_image(image_path):
    """
    Анализирует одно изображение и возвращает словарь с результатами.
    
    Args:
        image_path: Путь к изображению
        
    Returns:
        dict: Результаты анализа или None если ошибка
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, mask_black = cv2.threshold(gray_img, 80, 255, cv2.THRESH_BINARY_INV)
    
    mask_blue = cv2.inRange(hsv, (90, 40, 40), (130, 255, 255))
    mask_red1 = cv2.inRange(hsv, (0, 40, 40), (10, 255, 255))
    mask_red2 = cv2.inRange(hsv, (160, 40, 40), (180, 255, 255))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    # Суженный диапазон для зеленого (исключаем желтый)
    mask_green = cv2.inRange(hsv, (50, 130, 120), (100, 255, 255))
    
    kernel = np.ones((3, 3), np.uint8)
    mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)

    # --- Черный ---
    black_area = cv2.countNonZero(mask_black)
    if black_area > BLACK_AREA_FLOOR_THRESHOLD:
        black_status = "пол (много)"
    elif black_area < BLACK_AREA_MIN_THRESHOLD:
        black_status = "мало/нет"
    else:
        contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        is_line = False
        for cnt in contours:
            if cv2.contourArea(cnt) < 100: 
                continue
            bw, bh = cv2.minAreaRect(cnt)[1]
            if bw > 0 and bh > 0 and max(bw, bh) / min(bw, bh) > 5:
                is_line = True
                break
        black_status = "линия" if is_line else "мало/нет"
        
    # --- Анализ фигур ---
    def analyze_shapes(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        has_oval, has_rect = False, False
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300:
                continue
            
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = max(bw, bh) / min(bw, bh) if min(bw, bh) > 0 else 999
            
            if circularity > 0.6 and aspect < 2.5:
                has_oval = True
                
            if len(approx) == 4:
                hull_area = cv2.contourArea(cv2.convexHull(cnt))
                solidity = float(area) / hull_area if hull_area > 0 else 0
                if solidity > 0.8:
                    has_rect = True
                    
        return has_oval, has_rect

    blue_oval, blue_rect = analyze_shapes(mask_blue)
    red_oval, red_rect = analyze_shapes(mask_red)
    
    # --- Красная линия ---
    red_line = False
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_red:
        if cv2.contourArea(cnt) >= 300:
            bw, bh = cv2.minAreaRect(cnt)[1]
            if min(bw, bh) > 0 and max(bw, bh) / min(bw, bh) > 5:
                red_line = True
                break
            
    # --- Зеленый (только площадь) ---
    green_area = cv2.countNonZero(mask_green)
    
    return {
        "black_status": black_status,
        "black_area_px": black_area,
        
        "has_blue_oval": blue_oval,
        "has_blue_rectangle": blue_rect,
        "blue_area_px": cv2.countNonZero(mask_blue),
        
        "has_red_oval": red_oval,
        "has_red_rectangle": red_rect,
        "has_red_line": red_line,
        "red_area_px": cv2.countNonZero(mask_red),
        
        "has_green": green_area > GREEN_AREA_THRESHOLD,
        "green_area_px": green_area,
        "mask_green": mask_green
    }

def analyze_directory(output_dir="Output", save_masks=True):
    """
    Анализирует все изображения в директории.
    
    Args:
        output_dir: Директория с изображениями (1.jpg, 2.jpg, ...)
        save_masks: Сохранять ли маски зеленого
        
    Returns:
        dict: Результаты анализа
    """
    # Ищем файлы 1.jpg, 2.jpg, 3.jpg, 4.jpg
    image_files = []
    for i in range(1, 5):  # Проверяем до 8 частей
        img_path = os.path.join(output_dir, f"{i}.jpg")
        if os.path.exists(img_path):
            image_files.append(img_path)
    
    if not image_files:
        print(f"Изображения (1.jpg, 2.jpg, ...) в {output_dir} не найдены.")
        return {}
    
    # Создаем папку для масок если нужно
    if save_masks:
        masks_dir = os.path.join(output_dir, "green_masks")
        os.makedirs(masks_dir, exist_ok=True)
    
    results = {}
    
    for img_path in sorted(image_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0])):
        filename = os.path.basename(img_path)
        print(f"\nАнализируем: {filename}...")
        analysis = analyze_image(img_path)
        
        if analysis:
            # Извлекаем маску перед сохранением
            mask_green = analysis.pop("mask_green")
            
            # Сохраняем маску зеленого
            if save_masks:
                base_name = os.path.splitext(filename)[0]
                mask_path = os.path.join(masks_dir, f"{base_name}_green_mask.png")
                cv2.imwrite(mask_path, mask_green)
                print(f"  [GREEN] Площадь: {analysis['green_area_px']} | "
                      f"Порог: {GREEN_AREA_THRESHOLD} | "
                      f"Обнаружен: {'ДА' if analysis['has_green'] else 'НЕТ'}")
                print(f"  Маска сохранена: {mask_path}")
            
            results[filename] = analysis
    
    # Сохраняем результаты в JSON
    json_path = os.path.join(output_dir, "analysis_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"\nАнализ завершен. Результаты сохранены в {json_path}")
    return results

def main():
    """Основная функция для запуска из командной строки"""
    analyze_directory(output_dir="Output", save_masks=True)

if __name__ == "__main__":
    main()