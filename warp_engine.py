# warp_engine.py (CPU専用版)
import json
import numpy as np
import cv2

warp_cache = {}

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
def _log(msg, log_func=None):
    if log_func:
        try:
            log_func(msg)
        except Exception:
            print(msg)
    else:
        print(msg)

# ----------------------------------------------------------------------
# Perspective matrix
# ----------------------------------------------------------------------
def generate_perspective_matrix(src_size, dst_points):
    w, h = src_size
    src_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    dst_pts = np.array(dst_points, np.float32)
    return cv2.getPerspectiveTransform(src_pts, dst_pts)

# ----------------------------------------------------------------------
# Display map
# ----------------------------------------------------------------------
DISPLAY_TO_SIMSET = {}
try:
    with open("config/display_map.json", "r", encoding="utf-8") as f:
        DISPLAY_TO_SIMSET = json.load(f).get("display_map", {})
except Exception:
    DISPLAY_TO_SIMSET = {}

# ----------------------------------------------------------------------
# prepare_warp (CPU計算のみ)
# ----------------------------------------------------------------------
def prepare_warp(display_name, mode, src_size,
                 load_points_func=None,
                 log_func=None):

    display_name = (
        display_name.replace("(", "_")
                    .replace(")", "_")
                    .replace(" ", "_")
                    .replace("-", "_")
                    .replace("/", "_")
    )

    cache_key = (display_name, mode, src_size)
    if cache_key in warp_cache:
        return warp_cache[cache_key]

    w, h = src_size

    # ------------------------------------------------
    # load points
    # ------------------------------------------------
    pts = None
    if load_points_func:
        try:
            pts = load_points_func(display_name, mode)
        except Exception:
            pts = None

    # ------------------------------------------------
    # base grid
    # ------------------------------------------------
    xs = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    ys = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    grid_u = xs / max(w - 1, 1)
    grid_v = ys / max(h - 1, 1)

    # ------------------------------------------------
    # perspective
    # ------------------------------------------------
    if mode == "perspective" and pts is not None and len(pts) >= 4:
        grid_pts = np.array(pts[:4], np.float32)
        if grid_pts.max() <= 1.5:
            grid_pts[:, 0] *= (w - 1)
            grid_pts[:, 1] *= (h - 1)
        else:
            grid_pts[:, 0] *= w / 1920.0
            grid_pts[:, 1] *= h / 1080.0

        M = generate_perspective_matrix(src_size, grid_pts)
        grid_u = cv2.warpPerspective(grid_u, M, (w, h))
        grid_v = cv2.warpPerspective(grid_v, M, (w, h))
        grid_u = np.clip(grid_u, 0.0, 1.0)
        grid_v = np.clip(grid_v, 0.0, 1.0)
        valid_mask = np.ones((h, w), np.float32)
        warp_cache[cache_key] = (grid_u.astype(np.float32), grid_v.astype(np.float32), valid_mask)
        return warp_cache[cache_key]

    # ------------------------------------------------
    # warp_map
    # ------------------------------------------------
    if mode != "warp_map" or pts is None:
        valid_mask = np.ones((h, w), np.float32)
        warp_cache[cache_key] = (grid_u.astype(np.float32), grid_v.astype(np.float32), valid_mask)
        return warp_cache[cache_key]

    pts_np = np.asarray(pts, np.float32)
    if pts_np.max() <= 1.5:
        pts_np[:, 0] *= (w - 1)
        pts_np[:, 1] *= (h - 1)
    else:
        pts_np[:, 0] *= w / 1920.0
        pts_np[:, 1] *= h / 1080.0

    if len(pts_np) != 36:
        _log("[warp_map] require exactly 36 outer points", log_func)
        return None

    # ------------------------------
    # 外周分解（時計回り）
    # ------------------------------
    idx = 0
    top    = pts_np[idx:idx+10]; idx += 10
    right  = pts_np[idx:idx+8];  idx += 8
    bottom = pts_np[idx:idx+10]; idx += 10
    left   = pts_np[idx:idx+8]
    bottom = bottom[::-1]
    left   = left[::-1]

    # ------------------------------
    # 再サンプリング関数
    # ------------------------------
    def resample_curve(curve, n):
        d = np.linalg.norm(curve[1:] - curve[:-1], axis=1)
        s = np.concatenate([[0.0], np.cumsum(d)])
        t = np.linspace(0, s[-1], n)
        x = np.interp(t, s, curve[:, 0])
        y = np.interp(t, s, curve[:, 1])
        return np.stack([x, y], axis=1)

    top    = resample_curve(top,    10)
    right  = resample_curve(right,  10)
    bottom = resample_curve(bottom, 10)
    left   = resample_curve(left,   10)

    # ------------------------------
    # Coons Patch 内部グリッド生成 (8x8セル)
    # ------------------------------
    n_cells = 8
    grid = np.zeros((n_cells + 1, n_cells + 1, 2), np.float32)
    for j in range(n_cells + 1):
        v = j / n_cells
        for i in range(n_cells + 1):
            u = i / n_cells
            B = (1 - v) * top[i] + v * bottom[i] + (1 - u) * left[j] + u * right[j]
            C = (1 - u) * (1 - v) * top[0] + u * (1 - v) * top[-1] + (1 - u) * v * bottom[0] + u * v * bottom[-1]
            grid[j, i] = B - C

    # ------------------------------
    # セル単位で逆射影
    # ------------------------------
    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)
    filled = np.zeros((h, w), np.uint8)
    valid_mask = np.zeros((h, w), np.float32)

    for j in range(n_cells):
        for i in range(n_cells):
            dst = np.array([
                grid[j,   i],
                grid[j,   i+1],
                grid[j+1, i+1],
                grid[j+1, i]
            ], np.float32)

            src = np.array([
                [i/n_cells,   j/n_cells],
                [(i+1)/n_cells, j/n_cells],
                [(i+1)/n_cells, (j+1)/n_cells],
                [i/n_cells,   (j+1)/n_cells]
            ], np.float32)

            Hinv = cv2.getPerspectiveTransform(dst, src)
            cell_mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(cell_mask, [dst.astype(np.int32)], 1)
            ys, xs = np.where((cell_mask == 1) & (filled == 0))
            if len(xs) == 0:
                continue

            pts_xy = np.stack([xs, ys, np.ones_like(xs)], axis=1)
            uvw = (Hinv @ pts_xy.T).T
            valid = uvw[:, 2] != 0
            u = np.zeros(len(xs), np.float32)
            v = np.zeros(len(xs), np.float32)
            u[valid] = uvw[valid, 0] / uvw[valid, 2]
            v[valid] = uvw[valid, 1] / uvw[valid, 2]

            map_x[ys, xs] = np.clip(u, 0.0, 1.0)
            map_y[ys, xs] = np.clip(v, 0.0, 1.0)
            filled[ys, xs] = 1
            valid_mask[ys, xs] = 1.0

    warp_cache[cache_key] = (map_x, map_y, valid_mask)
    return warp_cache[cache_key]

# ----------------------------------------------------------------------
# warp_image
# ----------------------------------------------------------------------
def warp_image(image, map_x, map_y):
    return cv2.remap(
        image,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
