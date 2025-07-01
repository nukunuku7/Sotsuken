# warp_engine.py（初回のみ補正マップ作成、以降は再利用）

import os
import json
import numpy as np
import cv2
from grid_utils import sanitize_filename

SETTINGS_DIR = "settings"
BLEND_WIDTH_RATIO = 0.1

warp_cache = {}  # display_name → precomputed data

def get_points_path(display_name, mode):
    safe_name = sanitize_filename(display_name)
    return os.path.join(SETTINGS_DIR, f"{safe_name}_{mode}_points.json")

def load_points(display_name, mode):
    path = get_points_path(display_name, mode)
    if not os.path.exists(path):
        print(f"[DEBUG] グリッドファイルが存在しません: {path}")
        return None
    with open(path, "r") as f:
        points = json.load(f)
    print(f"[DEBUG] グリッド読み込み成功: {path} ({len(points)}点)")
    return np.array(points, dtype=np.float32)

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
    if len(dst_points) != 4:
        raise ValueError("射影変換には4点が必要です")
    src_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    dst_pts = np.array(dst_points, dtype=np.float32)
    return cv2.getPerspectiveTransform(src_pts, dst_pts)

def prepare_warp(display_name, mode, src_size):
    points = load_points(display_name, mode)
    if points is None or len(points) < 4:
        return None

    h, w = src_size
    if mode == "perspective":
        matrix = generate_perspective_matrix((h, w), points[:4])
        fade = generate_fade_mask(w, h)
        return {"mode": mode, "matrix": matrix, "fade": fade}

    elif mode == "warp_map":
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [points.astype(np.int32)], 255)
        map_x = np.full((h, w), -1, dtype=np.float32)
        map_y = np.full((h, w), -1, dtype=np.float32)
        ys, xs = np.where(mask == 255)
        map_x[ys, xs] = xs
        map_y[ys, xs] = ys
        map_x = np.clip(map_x, 0, w - 1)
        map_y = np.clip(map_y, 0, h - 1)
        fade = generate_fade_mask(w, h)
        return {"mode": mode, "map_x": map_x, "map_y": map_y, "fade": fade}

    else:
        print(f"[警告] 未知の補正モードです: {mode}")
        return None

def warp_image(image, warp_info):
    if image is None or warp_info is None:
        return image

    h, w = image.shape[:2]
    try:
        if warp_info["mode"] == "perspective":
            matrix = warp_info["matrix"]
            warped = cv2.warpPerspective(image, matrix, (w, h),
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        elif warp_info["mode"] == "warp_map":
            warped = cv2.remap(image, warp_info["map_x"], warp_info["map_y"],
                               interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        fade_mask = warp_info["fade"]
        for c in range(3):
            warped[:, :, c] = (warped[:, :, c].astype(np.float32) * fade_mask).astype(np.uint8)

        return warped

    except Exception as e:
        print(f"[エラー] warp_image失敗: {e}")
        return image
