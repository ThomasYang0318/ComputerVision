# FAST、pyramid、NMS
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
            val = int(image[py, px])
            brightness_values.append(val)
        else:
            brightness_values.append(0)
    
    center_value = int (image[p[1], p[0]])
    status_array = []

    for i in range(num_points):
        if brightness_values[i] >= center_value + t:
            status_array.append(1)
        elif brightness_values[i] <= center_value - t:
            status_array.append(-1)
        else:
            status_array.append(0)

    extended_array = np.concatenate([status_array, status_array])

    mask_pos = (extended_array == 1).astype(int) # astype(int) : True/False → 0/1
    mask_neg = (extended_array == -1).astype(int) 
    count_positive = np.convolve(mask_pos, np.ones(n), mode='valid') # np.convolve(要計算的資料, 權重（np.one(n) -> n個1）, valid:視窗完全覆蓋時)
    count_negative = np.convolve(mask_neg, np.ones(n), mode='valid')

    if np.any(count_positive == n) or np.any(count_negative == n):
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
            all_keypoints.append((x_original, y_original, level))

    return all_keypoints

def grid_NMS(all_keypoints, cell_size):
    grid = {}

    for x, y, level in all_keypoints:
        cell_x = x // cell_size
        cell_y = y // cell_size
        cell_coord = (cell_x, cell_y)

        if cell_coord not in grid:
            grid[cell_coord] = (x, y, level)  # 該區域第一次出現 → 存入

    final_keypoints = list(grid.values())
    return final_keypoints