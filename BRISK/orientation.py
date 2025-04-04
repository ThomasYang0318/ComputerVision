import numpy as np

# orientation 計算
def compute_orientation_with_pyramid(pyramid, keypoint, R, scale_factor):
    x0, y0, level = keypoint
    image = pyramid[level]

    M = 0.0  # 亮度加總
    mx = 0.0 
    my = 0.0 

    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            x = x0 + dx
            y = y0 + dy

            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                if dx ** 2 + dy ** 2 <= R ** 2:
                    I = int(image[y, x])
                    mx += I * dx
                    my += I * dy
                    M += I

    if M == 0:
        return 0.0  # 避免除以 0

    cx = mx / M
    cy = my / M
    theta = np.arctan2(cy, cx)  # 注意順序是 (y, x)
    return theta

def upgrade_keypoints_with_orientation(pyramid, original_keypoints, scale_factor, radius=5):
    upgraded = []
    num_levels = len(pyramid)

    for pt in original_keypoints:
        x, y = pt[:2]
        for level in range(num_levels):
            scale = scale_factor ** level
            x_scaled = int(x * scale)
            y_scaled = int(y * scale)
            image = pyramid[level]
            if 0 <= x_scaled < image.shape[1] and 0 <= y_scaled < image.shape[0]:
                angle = compute_orientation_with_pyramid(
                    pyramid, (x_scaled, y_scaled, level), radius, scale_factor
                )
                upgraded.append({
                    "x": x,
                    "y": y,
                    "angle": angle
                })
                break  # 找到第一個符合的層級就停止
    return upgraded
