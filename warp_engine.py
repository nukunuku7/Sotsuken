# warp_engine.py
# ------------------------------------------------------------
# Warp Engine (Cached Grid Warp)
#
# ・4点：射影変換（UV map生成）
# ・36点：10x10グリッド外周 → 内部補間
# ・歪み計算は初期化時に1回だけ（キャッシュ）
# ------------------------------------------------------------

import numpy as np
import cv2
from editor.grid_utils import load_points, get_virtual_id


# ------------------------------------------------------------
# internal cache
# ------------------------------------------------------------
_WARP_CACHE = {}


# ------------------------------------------------------------
# Perspective warp (4-point)
# ------------------------------------------------------------
def _build_perspective_map(points, width, height):
    if len(points) != 4:
        raise ValueError("Perspective mode requires exactly 4 points")

    # UIで指定された4点（投影先）
    dst = np.array(points, dtype=np.float32)

    # 元画像の矩形
    src = np.array(
        [
            [0, 0],
            [width, 0],
            [width, height],
            [0, height],
        ],
        dtype=np.float32,
    )

    # dst → src に戻す Homography
    H = cv2.getPerspectiveTransform(dst, src)

    # 全ピクセル座標
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )

    ones = np.ones_like(grid_x)
    coords = np.stack([grid_x, grid_y, ones], axis=-1)

    warped = coords @ H.T
    z = warped[..., 2:3]

    # 発散防止
    z[z == 0] = np.nan
    warped /= z

    map_x = warped[..., 0]
    map_y = warped[..., 1]

    return map_x, map_y


# ------------------------------------------------------------
# Boundary-only grid warp (36 points)
# ------------------------------------------------------------
def _build_boundary_grid_map(points, width, height):
    """
    points : 10x10グリッド外周のみ（36点、時計回り）
    """

    if len(points) != 36:
        raise ValueError("Boundary grid requires exactly 36 points")

    p = np.array(points, dtype=np.float32)
    grid = np.zeros((10, 10, 2), dtype=np.float32)

    idx = 0

    # 上辺
    for x in range(10):
        grid[0, x] = p[idx]
        idx += 1

    # 右辺
    for y in range(1, 9):
        grid[y, 9] = p[idx]
        idx += 1

    # 下辺
    for x in range(9, -1, -1):
        grid[9, x] = p[idx]
        idx += 1

    # 左辺
    for y in range(8, 0, -1):
        grid[y, 0] = p[idx]
        idx += 1

    # ---- 内部補間（bilinear）----
    for y in range(1, 9):
        fy = y / 9.0
        for x in range(1, 9):
            fx = x / 9.0

            top = (1 - fx) * grid[0, 0] + fx * grid[0, 9]
            bottom = (1 - fx) * grid[9, 0] + fx * grid[9, 9]
            grid[y, x] = (1 - fy) * top + fy * bottom

    # ---- フル解像度へ ----
    map_x = cv2.resize(
        grid[..., 0],
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    map_y = cv2.resize(
        grid[..., 1],
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    return map_x, map_y


# ------------------------------------------------------------
# prepare_warp（初回のみ計算）
# ------------------------------------------------------------
def prepare_warp(display_name, mode, src_size, log_func=None):
    width, height = src_size
    virt = get_virtual_id(display_name)

    cache_key = (virt, mode, width, height)
    if cache_key in _WARP_CACHE:
        return _WARP_CACHE[cache_key]

    points = load_points(display_name, mode)
    if points is None:
        if log_func:
            log_func(f"[warp] grid NOT FOUND ({virt}, {mode})")
        return None

    if log_func:
        log_func(f"[warp] building warp ({virt}, {mode})")

    if mode == "perspective":
        map_x, map_y = _build_perspective_map(points, width, height)
    elif mode in ("map", "warp_map"):
        map_x, map_y = _build_boundary_grid_map(points, width, height)
    else:
        raise RuntimeError(f"Unsupported warp mode: {mode}")

    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    _WARP_CACHE[cache_key] = (map_x, map_y)
    return map_x, map_y


# ------------------------------------------------------------
# GPU helper
# ------------------------------------------------------------
def convert_maps_to_uv_texture_data(map_x, map_y, width, height):
    u = map_x / float(width)
    v = map_y / float(height)

    uv = np.dstack((u, v)).astype(np.float32)
    return uv.tobytes()
