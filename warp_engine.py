import os
import json
import numpy as np
import cv2

# projector_profiles 配下を参照
BASE_DIR = os.path.join("config", "projector_profiles")
BLEND_WIDTH_RATIO = 0.1

warp_cache = {}  # display_name → precomputed data

def get_points_path(display_name, mode):
    """
    DISPLAY名から対応するJSONファイルパスを生成
    """
    # 既存のファイル名に合わせて "__._DISPLAY2" をそのまま使う
    safe_name = display_name.replace("\\", "_").replace(":", "_")
    path1 = os.path.join(BASE_DIR, f"{display_name}_{mode}_points.json")
    path2 = os.path.join(BASE_DIR, f"{safe_name}_{mode}_points.json")

    # 優先的に既存ファイルを探す
    for p in [path1, path2]:
        if os.path.exists(p):
            return p

    # 見つからなければ既存フォーマットに寄せる
    return os.path.join(BASE_DIR, f"__._{display_name}_{mode}_points.json")

def load_points(display_name, mode):
    path = get_points_path(display_name, mode)
    if not os.path.exists(path):
        print(f"[DEBUG] グリッドファイルが存在しません: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
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
    cache_key = (display_name, mode, src_size)
    if cache_key in warp_cache:
        return warp_cache[cache_key]

    points = load_points(display_name, mode)
    if points is None or len(points) < 4:
        return None

    h, w = src_size
    if mode == "perspective":
        matrix = generate_perspective_matrix((h, w), points[:4])
        fade = generate_fade_mask(w, h)
        warp_cache[cache_key] = {"mode": mode, "matrix": matrix, "fade": fade}
        return warp_cache[cache_key]

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
        warp_cache[cache_key] = {"mode": mode, "map_x": map_x, "map_y": map_y, "fade": fade}
        return warp_cache[cache_key]

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
