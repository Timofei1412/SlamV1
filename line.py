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
        y1 = constrain(self.x - self.size, 0, height)
        y2 = constrain(self.x + self.size, 0, height)

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
        y1 = constrain(self.x - self.size, 0, height)
        y2 = constrain(self.x + self.size, 0, height)

        if x2 <= x1 or y2 <= y1:
            self.value = 0
        else:
            self.value = int(gray[y1:y2, x1:x2].mean())
        return self.value

    def draw(self, img:np.ndarray, color:tuple = (0, 0, 255)):
        ul = (self.x - self.size, self.y - self.size)
        br = (self.x + self.size, self.y + self.size)
        cv2.rectangle(img, ul, br, color, 2)


class LineSensorModule:
    def __init__(self, x:int, y: int, rotation: bool = False, size:int = 20, spacing:int = 60, blackThresh:int = 120):
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

    def getFloorColor(self, bgr:np.ndarray)-> int:
        val = self.sensors[2].readBGR(bgr)
        if val < self.blackThresh:
            return 1
        return 0
    
    def calibrateBlackThresh(self):
        raise NotImplementedError
        def _mouse_callback(event, x, y, flags, param):
            """
            event: тип события (клик, движение и т.д.)
            x, y: координаты курсора
            flags: дополнительные флаги (Ctrl, Shift и т.д.)
            """
            if event == cv2.EVENT_LBUTTONDOWN:
                
            
            elif event == cv2.EVENT_RBUTTONDOWN:
                print(f'Правый клик в ({x}, {y})')
                cv2.rectangle(img, (x-10, y-10), (x+10, y+10), (0, 0, 255), -1)
            
            elif event == cv2.EVENT_MOUSEMOVE:
                print(f'Движение: ({x}, {y})')

    def getError(self, gray:np.ndarray)-> int:
        raise NotImplementedError
    
    def draw(self, img: np.ndarray, color:tuple = (0, 0, 255)):
        cv2.circle(img, self.center, 5, color, 3)
        for i in self.sensors:
            i.draw(img, color=color)
    

class LineController:
    def __init__(self, centerX: int, centerY: int, dist: int, size: int = 20, spacing: int = 60):
        self.center = (centerX, centerY)

        self.modules = [
            LineSensorModule(centerX - dist, centerY, rotation=True, size=size, spacing = spacing),
            LineSensorModule(centerX, centerY + dist, rotation=False, size=size, spacing = spacing),
            LineSensorModule(centerX + dist, centerY, rotation=True, size=size, spacing = spacing),
            LineSensorModule(centerX, centerY - dist, rotation=False, size=size, spacing= spacing),
        ]

    def getLineErrorAndCrossData(self, direction:int)->list[int, int]:
        raise NotImplementedError

    def draw(self, img:np.ndarray, color:tuple = (0, 0, 255)):
        cv2.circle(img, self.center, 5, color, 3)
        for i in self.modules:
            i.draw(img, color=color)


if __name__ == "__main__":
    img = cv2.imread("Images/img1.jpg")

    # s = Sensor(1000, 300, 15)
    # print(f"error = {s.readBGR(img)}")
    # s.draw(img)
    
    # module = LineSensorModule(1000, 300)
    # module2 = LineSensorModule(300, 1000, rotation=True)
    # module.draw(img)
    # module2.draw(img)
    
    l = LineController(1000, 1000, 500)

    l.draw(img)

    drawImageOnScreen("IMG", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

