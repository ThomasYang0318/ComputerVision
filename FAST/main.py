from detector import detect_corners_in_image
from pyramid import build_pyramid, detect_corners_in_pyramid
from nms import grid_NMS

import cv2
import os

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

    # 無金字塔版本
    for t in t_values:
        for n in n_values:
            keypoints = detect_corners_in_image(image, t, n)
            keypoints_nms = grid_NMS(keypoints, cell_size=3)
            color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            for (x, y) in keypoints_nms:
                cv2.circle(color_img, (x, y), 1, (0, 0, 255), -1)
            count = len(keypoints_nms)
            filename = f"{output_dir}/NoPyramid_t{t}_n{n}_{count}.png"
            cv2.imwrite(filename, color_img)
            print(f"Saved: {filename} ({count} pts)")

    # 有金字塔版本
    pyramid = build_pyramid(image, num_levels=4, scale_factor=0.5)
    for t in t_values:
        for n in n_values:
            keypoints = detect_corners_in_pyramid(pyramid, t, n, scale_factor=0.5)
            keypoints_nms = grid_NMS(keypoints, cell_size=3)
            color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            for (x, y) in keypoints_nms:
                cv2.circle(color_img, (x, y), 1, (0, 0, 255), -1)
            count = len(keypoints_nms)
            filename = f"{output_dir}/Pyramid_t{t}_n{n}_{count}.png"
            cv2.imwrite(filename, color_img)
            print(f"Saved: {filename} ({count} pts)")