import numpy as np
import cv2
import matplotlib.pyplot as plt

def draw_radial_pattern(image, center, config_list):
    """
    在 image 上繪製多層 radial pattern
    config_list: List of (num_lines, inner_radius, outer_radius, thickness)
    """
    for num_lines, r1, r2, thickness in config_list:
        for i in range(num_lines):
            theta = 2 * np.pi * i / num_lines
            x1 = int(center[0] + r1 * np.cos(theta))
            y1 = int(center[1] + r1 * np.sin(theta))
            x2 = int(center[0] + r2 * np.cos(theta))
            y2 = int(center[1] + r2 * np.sin(theta))
            cv2.line(image, (x1, y1), (x2, y2), 0, thickness)

def generate_brisk_vs_fast_test_image(
    image_size=512,
    bg_center=150,
    bg_edge=200,
    line_thickness=3,
    blur_sigma=3,
    filename='brisk_vs_fast_test.png'
):
    """
    生成一張 2D 幾何測試圖，用於對比 FAST / BRISK Detector 的能力差異。
    - 小正方形 (左上) 與 大正方形 (右上)
    - 小圓形 (模糊, 左下) 與 大圓形 (清晰, 右下)
    - 中央多尺度放射線
    - 平滑漸層背景
    """

    # ===== 建立平滑漸層背景 =====
    Y, X = np.ogrid[:image_size, :image_size]
    center = (image_size // 2, image_size // 2)
    distance = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    max_distance = np.sqrt(center[0]**2 + center[1]**2)
    # 漸層公式：bg_center + (bg_edge - bg_center) * (distance / max_distance)
    gradient = bg_center + (bg_edge - bg_center) * (distance / max_distance)
    image = gradient.astype(np.uint8)

    # ===== 畫「小正方形」(左上) =====
    # 大小視情況可調整
    cv2.rectangle(image, (55, 55), (105, 105), 0, line_thickness)

    # ===== 畫「大正方形」(右上) =====
    cv2.rectangle(image, (382, 30), (482, 130), 0, line_thickness)

    # ===== 小圓形(左下) + 模糊處理 =====
    # 1. 建立遮罩並畫一個白色填滿大一點的圓
    mask = np.zeros_like(image)
    cv2.circle(mask, (80, 432), 50, 255, -1)  

    # 2. 模糊遮罩（可調整 blur_sigma）
    mask_blur = cv2.GaussianBlur(mask, (0, 0), blur_sigma)

    # 3. 將模糊後的遮罩用來把原圖該區域壓暗 (乘以 1 - alpha)
    alpha = mask_blur.astype(float) / 255.0
    image = image.astype(float)
    image = image * (1.0 - alpha)
    image = image.astype(np.uint8)

    # ===== 大圓形(右下) =====
    cv2.circle(image, (432, 432), 50, 0, line_thickness)

    # ===== 中央多尺度放射線 =====
    pattern_config = [
        (16, 30, 50, 1),  # 內圈：細線
        (32, 70, 100, 2), # 中圈：中線
        (64, 110, 140, 3)  # 外圈：粗線
    ]
    draw_radial_pattern(image, center, pattern_config)

    # ===== 儲存圖片 =====
    cv2.imwrite(filename, image)

    # ===== 顯示結果 =====
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.title('BRISK vs FAST Test Image')
    plt.axis('off')
    plt.show()

# ===== 主程式測試 =====
if __name__ == "__main__":
    generate_brisk_vs_fast_test_image()
    print("已生成測試圖 brisk_vs_fast_test.png")
