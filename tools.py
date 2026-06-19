import cv2

def drawImageOnScreen(window_name, image):
    """
    Масштабирует изображение так, чтобы его наибольшее измерение 
    было равно 1000 пикселям, и выводит результат на экран.
    
    :param image: numpy array (результат cv2.imread)
    :param window_name: имя окна OpenCV
    """
    if image is None or image.size == 0:
        print("Ошибка: передано пустое изображение.")
        return

    # 1. Получаем размеры исходного изображения
    height, width = image.shape[:2]
    
    # 2. Находим максимальное измерение
    max_dim = max(height, width)
    
    # Защита от деления на ноль (на случай битого изображения 0x0)
    if max_dim == 0:
        print("Ошибка: изображение имеет нулевой размер.")
        return

    # 3. Вычисляем коэффициент масштабирования
    # Если max_dim > 1000, scale будет < 1 (уменьшение)
    # Если max_dim <= 1000, scale будет >= 1 (увеличение)
    target_max = 1000
    scale = target_max / max_dim
    
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # 4. Выбираем метод интерполяции
    # INTER_AREA дает лучшее качество при уменьшении
    # INTER_CUBIC дает лучшее качество при увеличении
    if scale < 1.0:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC
        
    # 5. Масштабируем изображение
    resized_image = cv2.resize(
        image, 
        (new_width, new_height), 
        interpolation=interpolation
    )
    
    cv2.imshow(window_name, resized_image)
