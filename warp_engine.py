# warp_engine.py（warp_map: -1領域の安全処理追加）
import os
import json
import numpy as np
import cv2
import re


SETTINGS_DIR = "C:/Users/vrlab/.vscode/nukunuku/Sotsuken/settings"
POINT_FILE_SUFFIX = "_points.json"

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
    if len(dst_points) != 4:
        raise ValueError("射影変換には4点が必要です")
    src_pts = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32)
    dst_pts = np.array(dst_points[:4], dtype=np.float32)
    return cv2.getPerspectiveTransform(src_pts, dst_pts)


def warp_image(image, display_name="default", mode="perspective"):
    h, w = image.shape[:2]
    points, mtime = load_points(display_name)
    if points is None or len(points) < 4:
        print(f"[警告] ポイントが不足しているため補正をスキップ ({display_name})")
        return image

    elif mode == "warp_map":
        try:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [points.astype(np.int32)], 255)

            # OpenCVのremap用に、マスク外は0,0にマッピングし、マスク内のみ元座標をセット
            map_x = np.zeros((h, w), dtype=np.float32)
            map_y = np.zeros((h, w), dtype=np.float32)
            ys, xs = np.where(mask == 255)
            map_x[ys, xs] = xs
            map_y[ys, xs] = ys

            # 範囲超過を防止
            map_x = np.clip(map_x, 0, w - 1)
            map_y = np.clip(map_y, 0, h - 1)

            # デバッグ出力（必要なら）
            # print(f"[DEBUG] shape: {map_x.shape}, dtype: {map_x.dtype}")
            # print(f"[DEBUG] min/max map_x: {np.min(map_x)}, {np.max(map_x)}")
            # print(f"[DEBUG] min/max map_y: {np.min(map_y)}, {np.max(map_y)}")
            # print(f"[DEBUG] nan in map_x: {np.isnan(map_x).any()}")
            # print(f"[DEBUG] nan in map_y: {np.isnan(map_y).any()}")
            # print(f"[DEBUG] map_x sample: {map_x[0,0]}, map_y sample: {map_y[0,0]}")

            warped = cv2.remap(image, map_x, map_y,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

            # ブレンディングマスク適用
            fade_mask = generate_fade_mask(w, h)
            for c in range(3):
                warped[:, :, c] = (warped[:, :, c].astype(np.float32) * fade_mask).astype(np.uint8)

            return warped

        except Exception as e:
            print(f"[エラー] warp_image失敗 ({display_name}): {e}")
            return image if 'image' in locals() else np.zeros((h, w, 3), dtype=np.uint8)
