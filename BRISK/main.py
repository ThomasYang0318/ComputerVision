import cv2
import numpy as np
import math
import os

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

def compute_orientation_with_pyramid(pyramid, keypoints, R, scale_factor):
    oriented_keypoints = []
    for (x_orig, y_orig, level) in keypoints:
        # 先將「原圖座標」轉為「對應層」的座標
        scale = scale_factor ** level
        x_l = int(x_orig * scale)
        y_l = int(y_orig * scale)

        # 使用該層影像來計算角度
        image = pyramid[level]
        M = mx = my = 0.0
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                if dx*dx + dy*dy <= R*R:
                    x_p = x_l + dx
                    y_p = y_l + dy
                    if 0 <= x_p < image.shape[1] and 0 <= y_p < image.shape[0]:
                        I = float(image[y_p, x_p])
                        mx += I * dx
                        my += I * dy
                        M += I
        if M == 0:
            angle = 0.0
        else:
            cx = mx / M
            cy = my / M
            angle = np.arctan2(cy, cx)

        oriented_keypoints.append({
            "x": x_orig,   # 保留原圖座標
            "y": y_orig,
            "angle": angle
        })
    return oriented_keypoints


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

# ===== Main =================================================================
if __name__ == "__main__":
    image_path = "../experiment_dataset/brisk_vs_fast_test.png"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("無法讀取圖片，請檢查路徑:", image_path)
        exit(1)

    t_values = [5, 10, 20]
    n_values = [9, 12]
    output_dir = "output_compare_pyramid"
    os.makedirs(output_dir, exist_ok=True)

    pyramid = build_pyramid(image, num_levels=4, scale_factor=0.5)
    for t in t_values:
        for n in n_values:
            keypoints = detect_corners_in_pyramid(pyramid, t, n, scale_factor=0.5)
            keypoints_nms = grid_NMS(keypoints, cell_size=3)
            # 畫角點帶方向
            keypoints_with_angle = compute_orientation_with_pyramid(pyramid, keypoints_nms, R=5, scale_factor=0.5)
            result_img = draw_keypoints_with_orientation(image, keypoints_with_angle)
            """ 畫角點
            color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            for (x, y) in keypoints_nms:
                cv2.circle(color_img, (x, y), 1, (0, 0, 255), -1)
            """
            count = len(keypoints_nms)
            filename = f"{output_dir}/output_with_orientation_t{t}_n{n}_{count}.png"
            cv2.imwrite(filename, result_img)
            print(f"Saved: {filename} ({count} pts)")