#!/usr/bin/env python3
import cv2
import numpy as np
from plane import remap_frame, build_combined_maps

# Ваши параметры
REMAPPING_PARAMS = dict(
    cx=308, cy=234, outer_r=230,
    lens_deg=-81.86, cone_power=2.245,
    rotation_deg=-2.0, top_size=640, field_scale=0.70,
    background=(0, 0, 0),
)

def process_frame_with_crosses(frame, map_x, map_y, background=(0, 0, 0)):
    """Обрабатывает один кадр: разворачивает через remap_frame и ищет перекрестки"""
    
    try:
        # Разворачиваем изображение используя remap_frame
        unwrapped_img = remap_frame(
            frame=frame,
            map_x=map_x,
            map_y=map_y,
            background_rgb=background,
            interpolation=cv2.INTER_LINEAR
        )
        
        if unwrapped_img is None or unwrapped_img.size == 0:
            return None, None
        
        # Обрабатываем развернутое изображение
        gray = cv2.cvtColor(unwrapped_img, cv2.COLOR_BGR2GRAY)
        
        # Инвертируем
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Скелетизация
        try:
            skeleton = cv2.ximgproc.thinning(thresh)
        except AttributeError:
            # Если ximgproc недоступен, используем морфологическую операцию
            skeleton = thresh.copy()
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            done = False
            while not done:
                eroded = cv2.erode(skeleton, element)
                temp = cv2.dilate(eroded, element)
                temp = cv2.subtract(skeleton, temp)
                skeleton = cv2.bitwise_or(skeleton, temp)
                done = cv2.countNonZero(skeleton) == cv2.countNonZero(eroded)
                skeleton = eroded.copy()
        
        # Ищем пересечения с более строгим порогом
        cross_kernel = np.array([[1,1,1],
                                 [1,0,1],
                                 [1,1,1]], dtype=np.uint8)
        
        conv = cv2.filter2D(skeleton.astype(np.uint8), -1, cross_kernel)
        
        # Увеличиваем порог с 3 до 4 или 5 для более строгого поиска
        intersections = np.where(conv >= 4)  # Было 3, теперь 4
        
        # Дополнительная фильтрация: убираем близкие точки
        if len(intersections[0]) > 0:
            points = list(zip(intersections[1], intersections[0]))  # (x, y)
            filtered_points = []
            
            for point in points:
                # Проверяем, не слишком ли близко к уже добавленным точкам
                too_close = False
                for existing_point in filtered_points:
                    distance = np.sqrt((point[0] - existing_point[0])**2 + 
                                     (point[1] - existing_point[1])**2)
                    if distance < 5:  # Минимальное расстояние между точками
                        too_close = True
                        break
                
                if not too_close:
                    filtered_points.append(point)
            
            # Рисуем только отфильтрованные перекрестки
            result_img = unwrapped_img.copy()
            for x, y in filtered_points:
                cv2.circle(result_img, (x, y), 3, (0, 255, 0), -1)
        else:
            result_img = unwrapped_img.copy()
        
        return result_img, skeleton
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None, None

def main():
    # Открываем видеофайл для получения размеров кадра
    video_path = 'Images/vid1.mp4'  # Укажите путь к вашему видео
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
    
    # Читаем первый кадр чтобы получить размеры
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame")
        cap.release()
        return
    
    # Возвращаемся к началу видео
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    height, width = first_frame.shape[:2]
    
    # Строим карты преобразования один раз (это экономит время)
    print("Building transformation maps...")
    map_x, map_y = build_combined_maps(
        top_size=REMAPPING_PARAMS['top_size'],
        source_width=width,
        source_height=height,
        cx=REMAPPING_PARAMS['cx'],
        cy=REMAPPING_PARAMS['cy'],
        outer_r=REMAPPING_PARAMS['outer_r'],
        rotation_deg=REMAPPING_PARAMS['rotation_deg'],
        field_scale=REMAPPING_PARAMS['field_scale'],
        lens_deg=REMAPPING_PARAMS['lens_deg'],
        cone_power=REMAPPING_PARAMS['cone_power'],
    )
    print("Maps built successfully!")
    
    # Создаем окна
    cv2.namedWindow('Result with Crosses', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Skeleton', cv2.WINDOW_NORMAL)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 10 == 0:  # Выводим прогресс каждые 10 кадров
            print(f"Processing frame {frame_count}...")
        
        # Обрабатываем кадр
        result_img, skeleton = process_frame_with_crosses(
            frame, 
            map_x, 
            map_y, 
            REMAPPING_PARAMS['background']
        )
        
        if result_img is not None and skeleton is not None:
            # Показываем результаты
            cv2.imshow('Result with Crosses', result_img)
            
            # Для скелета создаем цветное изображение для лучшей видимости
            skeleton_color = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
            cv2.imshow('Skeleton', skeleton_color)
        
        # Обработка клавиш
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('p'):  # Pause
            cv2.waitKey(0)
    
    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == "__main__":
    main()