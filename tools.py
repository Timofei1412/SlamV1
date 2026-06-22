# import cv2
# import numpy as np
# import logging


# def drawImageOnScreen(window_name:str, image:np.ndarray):
#     """
#     Масштабирует изображение так, чтобы его наибольшее измерение 
#     было равно 1000 пикселям, и выводит результат на экран.
    
#     :param image: numpy array (результат cv2.imread)
#     :param window_name: имя окна OpenCV
#     """
#     if image is None or image.size == 0:
#         print("Ошибка: передано пустое изображение.")
#         return

#     height, width = image.shape[:2]
    
#     max_dim = max(height, width)
    
#     if max_dim == 0:
#         print("Ошибка: изображение имеет нулевой размер.")
#         return

#     target_max = 1000
#     scale = target_max / max_dim
    
#     new_width = int(width * scale)
#     new_height = int(height * scale)
 
#     if scale < 1.0:
#         interpolation = cv2.INTER_AREA
#     else:
#         interpolation = cv2.INTER_CUBIC
        
#     resized_image = cv2.resize(
#         image, 
#         (new_width, new_height), 
#         interpolation=interpolation
#     )
    
#     cv2.imshow(window_name, resized_image)


# def constrain(val: int, minn: int, maxx: int) -> int:
#     return max(minn, min(val, maxx))

# def makeSize(data:str, length:int):
#     return data + "".join(["0" for i in range(length -len(data))])


# if __name__ == "__main__":
#     print(makeSize("Tima", 20))
