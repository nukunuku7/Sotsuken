# warp_engine.py（grid_utils統合＆安全性向上）
import os
import json
import numpy as np
import cv2
import re

from grid_utils import generate_perimeter_points, generate_perspective_points  # 共通化

SETTINGS_DIR = "C:/Users/vrlab/.vscode/nukunuku/Sotsuken/settings"
POINT_FILE_SUFFIX = "_points.json"
BLEND_WIDTH_RATIO = 0.1  # 横方向ブレンド率

def sanitize_filename(name):
    return re.sub(r'[\\/:*?"<>|]', '_', name)

def load_points(display_name):
    path = os.path.join(SETTINGS_DIR, f"{sanitize_filename(display_name)}{POINT_FILE_SUFFIX}")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        points = json.load(f)
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

def warp_image(image, display_name="default", mode="perspective"):
    if image is None:
        print(f"[エラー] 入力画像がNoneです ({display_name})")
        return None

    h, w = image.shape[:2]
    points = load_points(display_name)
    if points is None or len(points) < 4:
        print(f"[警告] 補正ポイントが不足しているためスキップします ({display_name})")
        return image

    try:
        if mode == "perspective":
            matrix = generate_perspective_matrix((h, w), points[:4])
            warped = cv2.warpPerspective(image, matrix, (w, h),
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        elif mode == "warp_map":
            # マスク作成
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [points.astype(np.int32)], 255)

            # remap用座標マップ（マスク外は無効領域）
            map_x = np.full((h, w), -1, dtype=np.float32)
            map_y = np.full((h, w), -1, dtype=np.float32)
            ys, xs = np.where(mask == 255)
            map_x[ys, xs] = xs
            map_y[ys, xs] = ys

            # 範囲外防止（OpenCV remap の仕様上、範囲内に収める）
            map_x = np.clip(map_x, 0, w - 1)
            map_y = np.clip(map_y, 0, h - 1)

            warped = cv2.remap(image, map_x, map_y,
                               interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        else:
            print(f"[警告] 未知の補正モードです: {mode}")
            return image

        # アルファブレンディング（左右端のフェード）
        fade_mask = generate_fade_mask(w, h)
        for c in range(3):
            warped[:, :, c] = (warped[:, :, c].astype(np.float32) * fade_mask).astype(np.uint8)

        return warped

    except Exception as e:
        print(f"[エラー] warp_image失敗 ({display_name}): {e}")
        return image
