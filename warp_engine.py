# warp_engine.py
import os
import json
import numpy as np
import cv2

# grid_utils から関数を利用
from editor.grid_utils import load_points, log

BLEND_WIDTH_RATIO = 0.1
warp_cache = {}  # display_name → precomputed data


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
    """
    display_name と mode から、warp に必要な情報を読み込む
    """
    cache_key = (display_name, mode, src_size)
    if cache_key in warp_cache:
        return warp_cache[cache_key]

    # --- grid_utils の load_points を使用 ---
    points = load_points(display_name, mode)
    if points is None or len(points) < 4:
        log(f"[WARN] グリッド点が不足または存在しません: {display_name} ({mode})")
        return None

    h, w = src_size
    if mode == "perspective":
        matrix = generate_perspective_matrix((h, w), points[:4])
        fade = generate_fade_mask(w, h)
        warp_cache[cache_key] = {"mode": mode, "matrix": matrix, "fade": fade}
        log(f"[OK] 射影変換情報を準備しました: {display_name}")
        return warp_cache[cache_key]

    elif mode == "warp_map":
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)

        map_x = np.full((h, w), -1, dtype=np.float32)
        map_y = np.full((h, w), -1, dtype=np.float32)
        ys, xs = np.where(mask == 255)
        map_x[ys, xs] = xs
        map_y[ys, xs] = ys
        map_x = np.clip(map_x, 0, w - 1)
        map_y = np.clip(map_y, 0, h - 1)
        fade = generate_fade_mask(w, h)

        warp_cache[cache_key] = {"mode": mode, "map_x": map_x, "map_y": map_y, "fade": fade}
        log(f"[OK] warp_map 情報を準備しました: {display_name}")
        return warp_cache[cache_key]

    else:
        log(f"[警告] 未知の補正モードです: {mode}")
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
        log(f"[ERROR] warp_image 失敗: {e}")
        return image
