# warp_engine.py（外周点によるwarp補正対応）
import os
import json
import numpy as np
import cv2
import re

SETTINGS_DIR = "C:/Users/vrlab/.vscode/nukunuku/Sotsuken/settings"
POINT_FILE_SUFFIX = "_points.json"

_map_cache = {}  # (h, w, display_name) をキーとした補正マップキャッシュ


def sanitize_filename(name):
    return re.sub(r'[\\/:*?"<>|]', '_', name)


def load_points(display_name):
    path = os.path.join(SETTINGS_DIR, f"{sanitize_filename(display_name)}{POINT_FILE_SUFFIX}")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return np.array(json.load(f), dtype=np.float32)


def generate_warp_map(h, w, display_name):
    key = (h, w, display_name)
    if key in _map_cache:
        return _map_cache[key]

    points = load_points(display_name)
    if points is None or len(points) < 4:
        print(f"[警告] ポイントが不足しているため歪み補正をスキップ ({display_name})")
        return None, None

    # 正規矩形への対応点作成（外周点に対応する等間隔グリッド）
    grid_points = points
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [grid_points.astype(np.int32)], 255)

    # 元画像の座標マップを作成
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0:
                map_x[y, x] = -1  # 黒塗り用に外に出す
                map_y[y, x] = -1
            else:
                map_x[y, x] = x
                map_y[y, x] = y

    _map_cache[key] = (map_x, map_y)
    return map_x, map_y


def warp_image(image, display_name="default"):
    h, w = image.shape[:2]
    map_x, map_y = generate_warp_map(h, w, display_name)
    if map_x is None:
        return image  # 補正なし
    warped = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    return warped
