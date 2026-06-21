import logging
import cv2
import numpy as np
from tools import drawImageOnScreen, constrain


class Sensor:
    """Один сенсор — считывает среднее значение n×n пикселей."""

    def __init__(self, x: int, y: int, size: int = 5):
        self.x = x
        self.y = y
        self.size = size
        self.value = 0
    
    def readGray(self, gray: np.ndarray) -> int:
        width, height = gray.shape[:2]

        x1 = constrain(self.x - self.size, 0, width)
        x2 = constrain(self.x + self.size, 0, width)
        y1 = constrain(self.y - self.size, 0, height)  # Исправлено self.x -> self.y
        y2 = constrain(self.y + self.size, 0, height)  # Исправлено self.x -> self.y

        if x2 <= x1 or y2 <= y1:
            self.value = 0
        else:
            self.value = int(gray[y1:y2, x1:x2].mean())
        return self.value
    
    def readBGR(self, img: np.ndarray) -> int:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        width, height = gray.shape[:2]

        x1 = constrain(self.x - self.size, 0, width)
        x2 = constrain(self.x + self.size, 0, width)
        y1 = constrain(self.y - self.size, 0, height)  # Исправлено self.x -> self.y
        y2 = constrain(self.y + self.size, 0, height)  # Исправлено self.x -> self.y

        if x2 <= x1 or y2 <= y1:
            self.value = 0
        else:
            self.value = int(gray[y1:y2, x1:x2].mean())
        return self.value

    def draw(self, img: np.ndarray, color: tuple = (0, 0, 255)):
        ul = (self.x - self.size, self.y - self.size)
        br = (self.x + self.size, self.y + self.size)
        cv2.rectangle(img, ul, br, color, 2)


class LineSensorModule:
    def __init__(self, x: int, y: int, rotation: bool = False, size: int = 20, spacing: int = 60, blackThresh: int = 120):
        self.center = (x, y)
        self.rotation = rotation

        xFactor = spacing * (1 if not rotation else 0)
        yFactor = spacing * (0 if not rotation else 1)

        self.sensors = [Sensor(x - xFactor, y - yFactor, size),
                        Sensor(x + xFactor, y + yFactor, size), 
                        Sensor(x - (3 * xFactor), y - (3 * yFactor), size)]
        
        self.blackThresh = blackThresh
        self.whiteVal = None
        self.blackVal = None

    def getFloorColor(self, bgr: np.ndarray) -> int:
        val = self.sensors[2].readBGR(bgr)
        return 1 if val < self.blackThresh else 0
    
    def calibrateBlackThresh(self, img: np.ndarray):
        """Калибровка порога. Кликните ЛКМ на белое, затем на черное."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        disp = img.copy()
        self.whiteVal, self.blackVal = None, None
        new_thresh = self.blackThresh

        def _mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                val = int(gray[y, x])
                if self.whiteVal is None:
                    self.whiteVal = val
                    print(f"[Калибровка] Белое значение: {val}")
                    cv2.circle(disp, (x, y), 5, (0, 255, 0), -1)
                elif self.blackVal is None:
                    self.blackVal = val
                    new_thresh = (self.whiteVal + val) // 2
                    print(f"[Калибровка] Черное значение: {val}")
                    print(f"[Калибровка] Новый blackThresh: {new_thresh}")
                    cv2.circle(disp, (x, y), 5, (0, 0, 255), -1)

        drawImageOnScreen("Calibration (White -> Black)", disp)
        cv2.setMouseCallback("Calibration (White -> Black)", _mouse_callback)
        
        while self.blackVal is None:
            if cv2.waitKey(20) == 27:  # Выход по ESC
                break
                
        cv2.destroyWindow("Calibration (White -> Black)")
        
        if self.blackVal is not None:
            self.blackThresh = new_thresh
            print(f"\n>>> Вставьте в __init__: blackThresh={new_thresh}\n")

    def getError(self, bgr: np.ndarray) -> int:
        floor = self.getFloorColor(bgr)
        vals = [i.readBGR(bgr)  for i in self.sensors[:1]]

        if floor == 0: #white bottom
            return vals[0] - vals[1]        
        #black bottom
        return vals[1] - vals[0]
    
    
    def draw(self, img: np.ndarray, color: tuple = (0, 0, 255)):
        cv2.circle(img, self.center, 5, color, 3)
        for i in self.sensors:
            i.draw(img, color=color)
    

class LineController:
    def __init__(self, centerX: int, centerY: int, dist: int, size: int = 20, spacing: int = 60):
        self.center = (centerX, centerY)
        self.dist = dist
        self.size = size
        self.spacing = spacing

        self.modules = [
            LineSensorModule(centerX, centerY - dist, rotation=False, size=size, spacing=spacing),
            LineSensorModule(centerX + dist, centerY, rotation=True, size=size, spacing=spacing),
            LineSensorModule(centerX, centerY + dist, rotation=False, size=size, spacing=spacing),
            LineSensorModule(centerX - dist, centerY, rotation=True, size=size, spacing=spacing),
        ]

        self.crossDetection = {
            "UP": [0, 1, 1, 0],
            "RIGHT": [1, 0, 0, 1],
            "DOWN": [0, 1, 1, 0],
            "LEN": [1, 0, 0, 1]
        }
        self.errorDetection = {
            "UP": [1, 0, 0, 0],
            "RIGHT": [0, 1, 0, 0],
            "DOWN": [0, 0, 1, 0],
            "LEFT": [0, 0, 0, 1]
        }
        
        self.directions = ["UP", "RIGHT", "DOWN", "LEFT"]


    def set_center(self, img: np.ndarray):
        """Позволяет выбрать новый центр мышкой на изображении. Выводит новые параметры в консоль."""
        disp = img.copy()
        self.draw(disp)  # рисуем текущее положение

        def _mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                conversionFactor = max(disp.shape[:2])//1000
                correction = 17 * conversionFactor

                self.set_center_manual(x * conversionFactor + correction, y * conversionFactor + correction)
                cv2.circle(disp, (x * conversionFactor + correction, y * conversionFactor + correction), 5, (0, 255, 0), -1)
                drawImageOnScreen("Set Center", disp)

        drawImageOnScreen("Set Center (click to choose center, ESC to cancel)", disp)
        cv2.setMouseCallback("Set Center (click to choose center, ESC to cancel)", _mouse_callback)

        while True:
            key = cv2.waitKey(20)
            if key == 27:  # ESC
                break
            # Если модули уже пересозданы (клик был), выходим
            if hasattr(self, '_center_set_done'):
                self._center_set_done = False
                break

        cv2.destroyWindow("Set Center (click to choose center, ESC to cancel)")

    def set_center_manual(self, new_centerX: int, new_centerY: int):
        """Перемещает центр контроллера вручную. Выводит новые параметры в консоль."""
        self.center = (new_centerX, new_centerY)
        self.modules = [
            LineSensorModule(new_centerX - self.dist, new_centerY, rotation=True, size=self.size, spacing=self.spacing),
            LineSensorModule(new_centerX, new_centerY + self.dist, rotation=False, size=self.size, spacing=self.spacing),
            LineSensorModule(new_centerX + self.dist, new_centerY, rotation=True, size=self.size, spacing=self.spacing),
            LineSensorModule(new_centerX, new_centerY - self.dist, rotation=False, size=self.size, spacing=self.spacing),
        ]
        print(f">>> Вставьте в __init__: LineController({new_centerX}, {new_centerY}, {self.dist}, size={self.size}, spacing={self.spacing})")
        self._center_set_done = True

    def getLineErrorAndCrossData(self, img: np.ndarray, direction: int) -> list[int, int]:
        dir = self.directions[direction]
        cross = 0
        for i in range(len(self.modules)):
            cross += self.modules[i].getError(img) * self.crossDetection[dir][i]
        
        error = 0
        for i in range(len(self.modules)):
            error += self.modules[i].getError(img) * self.errorDetection[dir][i]
        
        return [error, cross]
    
    def draw(self, img: np.ndarray, color: tuple = (0, 0, 255)):
        cv2.circle(img, self.center, 5, color, 3)
        for i in self.modules:
            i.draw(img, color=color)


if __name__ == "__main__":
    img = cv2.imread("Images/img1.jpg")
    
    l = LineController(1058, 820, 500, size=20, spacing=60)
    l.draw(img)

    drawImageOnScreen("IMG", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 