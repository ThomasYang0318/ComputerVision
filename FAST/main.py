import cv2
import numpy as np
import math

def is_corner(image, p, t, n):
    r = 3 # 半徑(論文設定的)
    num_points = 16 # 圓周點數(360度分16塊較合適)

    brightness_values = []

    for i in range(num_points):
        theta = math.radians((360 / num_points) * i)
        dx = round(r * math.cos(theta))
        dy = round(r * math.sin(theta))
        px = p[0] + dx
        py = p[1] + dy

        # 防止座標超出影像邊界
        if 0 <= py < image.shape[0] and 0 <= px < image.shape[1]:
            val = image[py, px]
            brightness_values.append(val)
        else:
            brightness_values.append(0)
    
    center_value = image[p[1], p[0]]
    status_array = []

    for i in range(num_points):
        if brightness_values[i] >= center_value + t:
            status_array.append(1)
        elif brightness_values[i] <= center_value - t:
            status_array.append(-1)
        else:
            status_array.append(0)

    extended_array = status_array + status_array

    count_positive = 0
    count_negative = 0

    for val in extended_array:
        if val == 1:
            count_positive += 1
            count_negative = 0
        elif val == -1:
            count_negative += 1
            count_positive = 0
        else:
            count_positive = 0
            count_negative = 0

        if count_positive >= n or count_negative >= n:
            return True
    

    return False

# 讀取圖片（灰階模式）
image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

# 轉成彩色，方便畫紅色圈圈
output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# 參數設定
t = 20  # 亮度差異閾值
n = 9 # 判斷角點條件

# 遍歷影像中每個像素
for y in range(3, image.shape[0] - 3):  # 留邊界
    for x in range(3, image.shape[1] - 3):
        if is_corner(image, (x, y), t, n):
            # 畫紅色小圈圈
            cv2.circle(output, (x, y), 1, (0, 0, 255), -1)

# 顯示結果
cv2.imshow('Corners', output)
cv2.waitKey(0)
cv2.destroyAllWindows()