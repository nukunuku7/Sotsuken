import cv2
import numpy as np

# キャッシュ用変数
_map_cache = {}

def warp_image(image):
    h, w = image.shape[:2]

    # キャッシュキー（画像サイズ）
    key = (h, w)
    if key not in _map_cache:
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
                new_r = r + 40 * np.sin(scale * np.pi / 2)
                if r != 0:
                    map_x[y, x] = cx + dx * new_r / r
                    map_y[y, x] = cy + dy * new_r / r
                else:
                    map_x[y, x] = x
                    map_y[y, x] = y

        _map_cache[key] = (map_x, map_y)

    map_x, map_y = _map_cache[key]
    warped = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    return warped
