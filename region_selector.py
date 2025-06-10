import cv2
import numpy as np

class RegionSelector:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.clone = self.image.copy()
        self.points = []

    def click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            cv2.circle(self.image, (x, y), 3, (0, 255, 0), -1)

    def run(self):
        self.image = self.clone.copy()
        self.points = []
        cv2.namedWindow("Selecciona puntos")
        cv2.setMouseCallback("Selecciona puntos", self.click)

        while True:
            cv2.imshow("Selecciona puntos", self.image)
            key = cv2.waitKey(1)
            if key == 13:  # ENTER
                break
            elif key == 27:  # ESC
                self.points = []
                break

        cv2.destroyWindow("Selecciona puntos")

        if len(self.points) >= 3:
            return np.array(self.points, dtype=np.int32)
        return None
