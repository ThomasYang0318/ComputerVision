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

def detect_corners_in_image(image, t, n):
    keypoints = []

    # 為了避免邊界問題，留 3 個像素的邊界
    for y in range(3, image.shape[0] - 3):
        for x in range(3, image.shape[1] - 3):
            if is_corner(image, (x, y), t, n):
                keypoints.append((x, y))

    return keypoints

def build_pyramid(image, num_levels, scale_factor):
    pyramid = [image]
    for i in range(1, num_levels):
        # 高斯模糊避免 aliasing
        blurred = cv2.GaussianBlur(pyramid[-1], (5, 5), 1)
        # 縮小影像
        small = cv2.resize(blurred, (0, 0), fx=scale_factor, fy=scale_factor)
        # 最小影像尺寸
        if small.shape[0] < 16 or small.shape[1] < 16:
            break
        pyramid.append(small)
    return pyramid

def detect_corners_in_pyramid(pyramid, t, n, scale_factor):
    all_keypoints = []

    # 針對每一層做 is_corner() 檢測
    for level , image in enumerate(pyramid):
        # 計算當前層的 scale
        scale = scale_factor ** level

        # 找出該層的角點
        keypoints = detect_corners_in_image(image, t, n)

        # 並把特徵點座標轉回原圖座標
        for x, y in keypoints:
            x_original = int(x / scale)
            y_original = int(y / scale)
            all_keypoints.append((x_original, y_original))

    return all_keypoints

# 讀取圖片（灰階模式）
image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

# Scale-Space參數設定 
num_levels = 4      # num_levels: 金字塔層數 
scale_factor = 0.5  # scale_factor: 縮小倍率
pyramid = build_pyramid(image, num_levels, scale_factor)

# 角點參數設定 
t = 20 # t: 亮度差異閾值
n = 9 # n: 判斷角點條件
keypoints = detect_corners_in_pyramid(pyramid, t, n, scale_factor)

# 轉成彩色，方便畫紅色圈圈(0, 0, 255)(B, G, R)
output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for x, y in keypoints:
    cv2.circle(output, (x, y), 1, (0, 0, 255), -1)

cv2.imshow('Pyramid Corners', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

