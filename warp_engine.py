# warp_engine.py
import cv2
import numpy as np

# 例：中央から球面風にゆがみ補正をかける
def warp_image(image):
    h, w = image.shape[:2]
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    cx, cy = w // 2, h // 2
    radius = min(cx, cy)

    for y in range(h):
        for x in range(w):
            dx = x - cx
            dy = y - cy
            r = np.sqrt(dx**2 + dy**2)
            if r == 0:
                scale = 1
            else:
                scale = r / radius
            new_r = r + 40 * np.sin(scale * np.pi / 2)  # ゆがみ強度
            if r != 0:
                map_x[y, x] = cx + dx * new_r / r
                map_y[y, x] = cy + dy * new_r / r
            else:
                map_x[y, x] = x
                map_y[y, x] = y

    warped = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    return warped
