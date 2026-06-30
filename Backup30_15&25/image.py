'''
Развертка конического изображения в панораму и разбиение ее на сектора.
Сектора сохраняются как 1.png, 2.png, 3.png, 4.png в папке Output
'''

import cv2
import numpy as np
import sys
import os
from tools import drawImageOnScreen

MANUAL_RADIUS = 778
PANORAMA_WIDTH = 300
DEBUG = False


def select_center_manually(image):
    """
    Позволяет выбрать центр зеркала вручную.
    По нажатию правой кнопки мыши (ЛКМ) фиксирует координаты.
    """
    window_name = "Select Center (ПКМ - выбрать, ESC - отмена)"
    h, w = image.shape[:2]
    max_dim = max(h, w)
    scale = 1.0
    if max_dim > 1000:
        scale = 1000.0 / max_dim
        
    scaled_img = cv2.resize(image, (int(w * scale), int(h * scale)))
    
    scale_x = image.shape[1] / scaled_img.shape[1]
    scale_y = image.shape[0] / scaled_img.shape[0]
    
    center = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal center
        if event == cv2.EVENT_LBUTTONDOWN:
            orig_x = int(x * scale_x)
            orig_y = int(y * scale_y)
            center = (orig_x, orig_y)
            print(f"ПКМ нажат. Выбран центр (координаты оригинала): {center}")
            cv2.destroyAllWindows()
    
    cv2.imshow(window_name, scaled_img)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    while center is None:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("Выбор центра отменен.")
            break
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
            
    cv2.destroyAllWindows()
    return center

def find_mirror_center_and_radius(image, manual_center=None, manual_radius=None):
    """
    Находит центр и радиус конического зеркала на изображении.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    center = None
    radius = 0
    
    if manual_center is None or manual_radius is None:
        circles = cv2.HoughCircles(
            gray, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=gray.shape[0]/2,
            param1=50, 
            param2=30, 
            minRadius=50, 
            maxRadius=0
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            largest_circle = max(circles, key=lambda c: c[2])
            center = (largest_circle[0], largest_circle[1])
            radius = largest_circle[2]
            print(f"Автоматически найдено: Центр {center}, Радиус {radius}")
        else:
            print("Автоматический поиск не удался. Используйте ручные параметры.")
    
    if manual_center is not None:
        center = manual_center
    if manual_radius is not None:
        radius = manual_radius
        
    if center is None or radius == 0:
        raise ValueError("Не удалось определить центр или радиус зеркала.")
        
    return center, radius

def unwrap_cone_image(image_path, output_path, manual_center=None, manual_radius=None, output_width=1500, debug = False):
    """
    Разворачивает изображение с конического зеркала в панораму.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}")
        return None
    
    center, max_radius = find_mirror_center_and_radius(
        img, manual_center, manual_radius
    )
    
    output_height = int(max_radius)
    dsize = (output_width, output_height)
    
    unwrapped_img = cv2.warpPolar(
        img, 
        dsize, 
        center, 
        max_radius, 
        flags=cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS
    )
    
    unwrapped_img = cv2.rotate(unwrapped_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, unwrapped_img)
    print(f"Развернутое изображение сохранено в {output_path}")
    if debug:
        cv2.circle(img, center, 5, (0, 0, 255), 5)
        drawImageOnScreen("Original with Center", img)
        
        cv2.imshow("Unwrapped Panorama", unwrapped_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return unwrapped_img

def select_and_extract_parts(unwrapped_img, output_dir, DEBUG):
    """
    Позволяет выбрать 8 точек для разделения панорамы на 4 части.
    Сохраняет части как 1.jpg, 2.jpg, 3.jpg, 4.jpg.
    Поддерживает склейку краев (циклический перенос).
    """
    window_name = "Select 8 points (Pairs: 1-2, 3-4, 5-6, 7-8). Enter=Save, R=Reset, Esc=Cancel"
    h, w = unwrapped_img.shape[:2]
    
    # Масштабирование до 1000 пикселей по максимальной стороне
    max_dim = max(h, w)
    scale = 1.0
    if max_dim > 1000:
        scale = 1000.0 / max_dim
        
    display_w = int(w * scale)
    display_h = int(h * scale)
    display_img = cv2.resize(unwrapped_img, (display_w, display_h))
    
    if DEBUG:
        points = []  # Координаты X в оригинальном размере
        
        def draw_current_state():
            out = display_img.copy()
            # Цвета для 4 частей: Красный, Зеленый, Синий, Желтый
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
            overlay = display_img.copy()
            
            # Рисуем вертикальные линии и номера точек
            for i in range(len(points)):
                x_orig = points[i]
                x_disp = int(x_orig * scale)
                cv2.line(out, (x_disp, 0), (x_disp, display_h), (255, 255, 255), 2)
                cv2.putText(out, str(i+1), (x_disp + 5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                
            # Подсвечиваем выбранные области (парами)
            for i in range(0, len(points) - len(points)%2, 2):
                x1_orig = points[i]
                x2_orig = points[i+1]
                color = colors[(i//2) % 4]
                
                x1_disp = int(x1_orig * scale)
                x2_disp = int(x2_orig * scale)
                
                if x1_disp < x2_disp:
                    # Обычная область
                    cv2.rectangle(overlay, (x1_disp, 0), (x2_disp, display_h), color, -1)
                else:
                    # Область переходит через границу (склеиваем концы)
                    cv2.rectangle(overlay, (x1_disp, 0), (display_w, display_h), color, -1)
                    cv2.rectangle(overlay, (0, 0), (x2_disp, display_h), color, -1)
                    
            # Полупрозрачное наложение
            cv2.addWeighted(overlay, 0.4, out, 0.6, 0, out)
            
            if len(points) == 8:
                cv2.putText(out, "ENTER: Save | R: Reset | ESC: Cancel", 
                            (10, display_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                cv2.putText(out, f"Points selected: {len(points)}/8", 
                            (10, display_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                            
            return out
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal points
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(points) < 8:
                    # Переводим координату клика обратно к оригинальному разрешению
                    orig_x = int(x / scale)
                    points.append(orig_x)
                    print(f"Точка {len(points)} выбрана: x={orig_x}")

                    cv2.imshow(window_name, draw_current_state())
                    
        cv2.imshow(window_name, draw_current_state())
        cv2.setMouseCallback(window_name, mouse_callback)
        
        confirmed = False
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == 27:  # Esc
                print("Выбор частей отменен.")
                cv2.destroyAllWindows()
                return
            elif key == 13 or key == 32:  # Enter or Space
                if len(points) == 8:
                    confirmed = True
                    print(f"points = [", end = "")
                    for i in points:
                        print(i, end = ", ")
                    print("]")
                    break
            elif key == ord('r') or key == 8:  # r or Backspace
                points = []
                print("Точки сброшены.")
                cv2.imshow(window_name, draw_current_state())
                
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except:
                break
        
        cv2.destroyAllWindows()
    else:
        points = [112, 263, 328, 479, 517, 669, 723, 80, ]
        confirmed = True

    if confirmed:
        os.makedirs(output_dir, exist_ok=True)
        for i in range(4):
            x1 = points[2*i]
            x2 = points[2*i+1]
            
            if x1 < x2:
                part = unwrapped_img[:, x1:x2]
            else:
                # Склеиваем концы изображения
                right_part = unwrapped_img[:, x1:]
                left_part = unwrapped_img[:, :x2]
                part = np.hstack((right_part, left_part))
                
            # Имена файлов: 1.jpg, 2.jpg, 3.jpg, 4.jpg
            part_path = os.path.join(output_dir, f"{i+1}.jpg")
            cv2.imwrite(part_path, part)
            print(f"Сохранена часть {i+1} в {part_path}")

if __name__ == "__main__":
    INPUT_IMAGE = "Images/photo3.jpg"           # Путь к исходному изображению с камеры
    OUTPUT_IMAGE = "Output/unwrapped_output.jpg"  # Путь для сохранения панорамы
    
    
    # Читаем изображение для интерактивного выбора центра
    img_for_selection = cv2.imread(INPUT_IMAGE)
    
    if img_for_selection is None:
        print(f"Ошибка: не удалось загрузить изображение {INPUT_IMAGE} для выбора центра.")
        sys.exit(1)
    
    if DEBUG:
        # Интерактивный выбор центра (ПКМ для выбора точки)
        MANUAL_CENTER = select_center_manually(img_for_selection)
        print(f"MANUAL_CENTER = ({MANUAL_CENTER[0]}, {MANUAL_CENTER[1]})")
    else:
        MANUAL_CENTER = (1143, 922)
    
    if MANUAL_CENTER is None:
        print("Центр не выбран. Завершение работы.")
        sys.exit(0)
    
    # Разворачиваем изображение
    unwrapped_img = unwrap_cone_image(
        image_path=INPUT_IMAGE,
        output_path=OUTPUT_IMAGE,
        manual_center=MANUAL_CENTER,
        manual_radius=MANUAL_RADIUS,
        output_width=PANORAMA_WIDTH,
        debug = DEBUG
    )
    
    # Если развертка успешна, предлагаем выбрать части
    if unwrapped_img is not None:
        output_dir = os.path.dirname(OUTPUT_IMAGE)
        select_and_extract_parts(unwrapped_img, output_dir, DEBUG)