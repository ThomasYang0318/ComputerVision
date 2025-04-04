import cv2
import numpy as np

def draw_keypoints_with_orientation(image, keypoints, color_point=(0, 0, 255), color_arrow=(255, 0, 0), length=10):
    # 若輸入是灰階，轉成彩色
    if len(image.shape) == 2 or image.shape[2] == 1:
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_color = image.copy()

    for kp in keypoints:
        x, y, angle = kp["x"], kp["y"], kp["angle"]
        end_x = int(x + length * np.cos(angle))
        end_y = int(y + length * np.sin(angle))
        cv2.circle(image_color, (x, y), 2, color_point, -1)
        cv2.arrowedLine(image_color, (x, y), (end_x, end_y), color_arrow, 1, tipLength=0.3)

    return image_color