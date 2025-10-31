import os
import json
import numpy as np
import cv2
from editor.grid_utils import load_points, log

warp_cache = {}

# --- 射影行列生成 ---
def generate_perspective_matrix(src_size, dst_points):
    w, h = src_size
    if len(dst_points) != 4:
        raise ValueError("射影変換には4点が必要です")
    src_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    dst_pts = np.array(dst_points, dtype=np.float32)
    return cv2.getPerspectiveTransform(src_pts, dst_pts)

# --- warp 準備 ---
def prepare_warp(display_name, mode, src_size):
    cache_key = (display_name, mode, src_size)
    if cache_key in warp_cache:
        return warp_cache[cache_key]

    points = load_points(display_name, mode)
    if points is None or len(points) < 4:
        log(f"[WARN] グリッド点が不足または存在しません: {display_name} ({mode})")
        return None

    pts = np.array(points, dtype=np.float32)
    w, h = src_size

    if mode == "perspective":
        matrix = generate_perspective_matrix((w, h), pts[:4])
        warp_cache[cache_key] = {"mode": mode, "matrix": matrix}
        log(f"[OK] 射影変換を準備: {display_name}")
        return warp_cache[cache_key]

    elif mode == "warp_map":
        ys, xs = np.indices((h, w), dtype=np.float32)
        map_x = xs.copy()
        map_y = ys.copy()
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
        outside = (mask == 0)
        map_x[outside] = 0.0
        map_y[outside] = 0.0
        warp_cache[cache_key] = {"mode": mode, "map_x": map_x, "map_y": map_y}
        log(f"[OK] warp_map を準備: {display_name}")
        return warp_cache[cache_key]

    return None

# --- warp 適用 ---
def warp_image(image, warp_info):
    if image is None or warp_info is None:
        return image

    h, w = image.shape[:2]
    try:
        if warp_info["mode"] == "perspective":
            warped = cv2.warpPerspective(image, warp_info["matrix"], (w, h),
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        elif warp_info["mode"] == "warp_map":
            warped = cv2.remap(image, warp_info["map_x"], warp_info["map_y"],
                               interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        else:
            return image

        return warped

    except Exception as e:
        log(f"[ERROR] warp_image 失敗: {e}")
        return image
