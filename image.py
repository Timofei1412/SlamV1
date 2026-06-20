import cv2
import numpy as np
import sys
from tools import drawImageOnScreen

def find_mirror_center_and_radius(image, manual_center=None, manual_radius=None):
    """
    Находит центр и радиус конического зеркала на изображении.
    Если автоматическое определение не удается, использует ручные значения.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    
    center = None
    radius = 0
    
    if manual_center is None or manual_radius is None:
        # Автоматический поиск окружности с помощью преобразования Хафа
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
            # Берем самую большую окружность (внешнюю границу зеркала)
            largest_circle = max(circles, key=lambda c: c[2])
            center = (largest_circle[0], largest_circle[1])
            radius = largest_circle[2]
            print(f"Автоматически найдено: Центр {center}, Радиус {radius}")
        else:
            print("Автоматический поиск не удался. Используйте ручные параметры.")

    # Переопределение ручными значениями, если они заданы или автопоиск не сработал
    if manual_center is not None:
        center = manual_center
    if manual_radius is not None:
        radius = manual_radius
        
    if center is None or radius == 0:
        raise ValueError("Не удалось определить центр или радиус зеркала.")
        
    return center, radius

def unwrap_cone_image(image_path, output_path, 
                      manual_center=None, manual_radius=None,
                      output_width=1500):
    """
    Разворачивает изображение с конического зеркала в панораму.
    Сначала поворачивает на 90 градусов, затем масштабирует до нужной ширины.
    """
    # 1. Чтение изображения
    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}")
        return

    # 2. Поиск геометрических параметров зеркала
    center, max_radius = find_mirror_center_and_radius(
        img, manual_center, manual_radius
    )

    # 3. Расчет параметров для полярной развертки
    # Высота выходного изображения равна радиусу
    output_height = int(max_radius)
    dsize = (output_width, output_height)

    # 4. Полярная развертка (Polar to Cartesian)
    unwrapped_img = cv2.warpPolar(
        img, 
        dsize, 
        center, 
        max_radius, 
        flags=cv2.WARP_POLAR_NEAR + cv2.WARP_FILL_OUTLIERS
    )
    
    unwrapped_img = cv2.rotate(unwrapped_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # 7. Сохранение результата
    cv2.imwrite(output_path, unwrapped_img)
    print(f"Развернутое изображение сохранено в {output_path}")

    # Отображение для проверки
    drawImageOnScreen("Original with Center", img)
    
    cv2.imshow("Unwrapped Panorama", unwrapped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# if __name__ == "__main__":
#     INPUT_IMAGE = "Images/img1.jpg"      # Путь к исходному изображению с камеры
#     OUTPUT_IMAGE = "Output/unwrapped_output.jpg" # Путь для сохранения панорамы
    
#     MANUAL_CENTER = (1071, 829)  # Пример: (640, 480)
#     MANUAL_RADIUS = 778  # Пример: 400
    
#     PANORAMA_WIDTH = 300 
    
#     ROTATION = 0 
#     # ----------------
#     for i in range(-10, 10):
#         unwrap_cone_image(
#             image_path=INPUT_IMAGE,
#             output_path=OUTPUT_IMAGE,
#             manual_center=(1071 + i, 829),
#             manual_radius=MANUAL_RADIUS,
#             output_width=PANORAMA_WIDTH,
#         )



#     #  unwrap_cone_image(
#     #         image_path=INPUT_IMAGE,
#     #         output_path=OUTPUT_IMAGE,
#     #         manual_center=MANUAL_CENTER,
#     #         manual_radius=MANUAL_RADIUS,
#     #         output_width=PANORAMA_WIDTH,
#     #     )


