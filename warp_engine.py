# warp_engine.py（JSONの変更を監視し、キャッシュを自動更新）
import os
import json
import numpy as np
import cv2
import re
import time

SETTINGS_DIR = "C:/Users/vrlab/.vscode/nukunuku/Sotsuken/settings"
POINT_FILE_SUFFIX = "_points.json"

_map_cache = {}  # key: (h, w, display_name) → (map_x, map_y, last_mtime)


def sanitize_filename(name):
    return re.sub(r'[\\/:*?"<>|]', '_', name)


def load_points(display_name):
    path = os.path.join(SETTINGS_DIR, f"{sanitize_filename(display_name)}{POINT_FILE_SUFFIX}")
    if not os.path.exists(path):
        return None, None
    with open(path, "r") as f:
        points = json.load(f)
    mtime = os.path.getmtime(path)
    return np.array(points, dtype=np.float32), mtime


def generate_warp_map(h, w, display_name):
    key = (h, w, display_name)
    points, mtime = load_points(display_name)
    if points is None or len(points) < 4:
        print(f"[警告] ポイントが不足しているため補正をスキップ ({display_name})")
        return None, None

    if key in _map_cache:
        cached_map_x, cached_map_y, cached_time = _map_cache[key]
        if mtime == cached_time:
            return cached_map_x, cached_map_y  # キャッシュが有効

    # 外周マスク作成
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [points.astype(np.int32)], 255)

    map_x = np.full((h, w), -1, dtype=np.float32)
    map_y = np.full((h, w), -1, dtype=np.float32)
    for y in range(h):
        for x in range(w):
            if mask[y, x] == 255:
                map_x[y, x] = x
                map_y[y, x] = y

    _map_cache[key] = (map_x, map_y, mtime)
    return map_x, map_y


def warp_image(image, display_name="default"):
    h, w = image.shape[:2]
    map_x, map_y = generate_warp_map(h, w, display_name)
    if map_x is None:
        return image
    warped = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    return warped