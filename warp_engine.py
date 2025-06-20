# warp_engine.py（射影変換モード対応）
import os
import json
import numpy as np
import cv2
import re
import time

SETTINGS_DIR = "C:/Users/vrlab/.vscode/nukunuku/Sotsuken/settings"
POINT_FILE_SUFFIX = "_points.json"

_map_cache = {}  # key: (h, w, display_name) → (map_x, map_y, last_mtime)
_mode_cache = {}  # optional: remember chosen mode

BLEND_WIDTH_RATIO = 0.1  # 横方向ブレンド率


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


def generate_fade_mask(w, h):
    fade = np.ones((h, w), dtype=np.float32)
    blend_w = int(w * BLEND_WIDTH_RATIO)
    for x in range(blend_w):
        alpha = x / blend_w
        fade[:, x] *= alpha
        fade[:, w - 1 - x] *= alpha
    return fade


def generate_perspective_matrix(src_size, dst_points):
    h, w = src_size
    src_pts = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32)
    dst_pts = np.array(dst_points[:4], dtype=np.float32)  # 左上→右上→右下→左下 の4点
    return cv2.getPerspectiveTransform(src_pts, dst_pts)


def warp_image(image, display_name="default", mode="perspective"):
    h, w = image.shape[:2]
    points, mtime = load_points(display_name)
    if points is None or len(points) < 4:
        print(f"[警告] ポイントが不足しているため補正をスキップ ({display_name})")
        return image

    if mode == "perspective":
        try:
            matrix = generate_perspective_matrix((h, w), points)
            warped = cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        except Exception as e:
            print(f"[射影変換エラー] {display_name}: {e}")
            return image
    else:
        # fallback: mask-based warp (旧方式)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [points.astype(np.int32)], 255)
        map_x = np.full((h, w), -1, dtype=np.float32)
        map_y = np.full((h, w), -1, dtype=np.float32)
        for y in range(h):
            for x in range(w):
                if mask[y, x] == 255:
                    map_x[y, x] = x
                    map_y[y, x] = y
        warped = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # ブレンディングマスク適用
    fade_mask = generate_fade_mask(w, h)
    for c in range(3):
        warped[:, :, c] = (warped[:, :, c].astype(np.float32) * fade_mask).astype(np.uint8)

    return warped
