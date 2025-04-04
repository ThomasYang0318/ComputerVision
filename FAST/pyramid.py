import cv2

from detector import detect_corners_in_image

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