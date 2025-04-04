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

    extended_array = np.concatenate([status_array, status_array])
    """
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
    """
    # 加速
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
