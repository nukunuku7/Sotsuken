# warp_engine.py
# ------------------------------------------------------------
# Warp Engine (Boundary-only Grid Warp)
#
# ・4点：射影変換
# ・36点：10x10グリッドの外周点のみ → 内部補間
# ・warp map は初期化時に1回だけ生成
# ------------------------------------------------------------

import numpy as np
import cv2
from editor.grid_utils import load_points, log


# ------------------------------------------------------------
# Perspective warp (4-point)
# ------------------------------------------------------------
def _build_perspective_map(points, width, height):
    if len(points) != 4:
        raise ValueError("Perspective mode requires exactly 4 points")

    dst = np.array(points, dtype=np.float32)

    src = np.array(
        [
            [0, 0],
            [width, 0],
            [width, height],
            [0, height],
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(dst, src)

    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )

    coords = np.stack(
        [grid_x, grid_y, np.ones_like(grid_x)], axis=-1
    )

    warped = coords @ M.T
    warped /= warped[..., 2:3]

    return warped[..., 0], warped[..., 1]


# ------------------------------------------------------------
# Boundary-only grid warp (36 points)
# ------------------------------------------------------------
def _build_boundary_grid_map(points, width, height):
    """
    points : 外周36点（10x10 gridの境界、時計回り）
    """

    if len(points) != 36:
        raise ValueError("Boundary grid requires exactly 36 points")

    # 10x10 grid を用意
    grid = np.zeros((10, 10, 2), dtype=np.float32)

    p = np.array(points, dtype=np.float32)

    idx = 0

    # 上辺 (0,0) → (0,9)
    for x in range(10):
        grid[0, x] = p[idx]
        idx += 1

    # 右辺 (1,9) → (8,9)
    for y in range(1, 9):
        grid[y, 9] = p[idx]
        idx += 1

    # 下辺 (9,9) → (9,0)
    for x in range(9, -1, -1):
        grid[9, x] = p[idx]
        idx += 1

    # 左辺 (8,0) → (1,0)
    for y in range(8, 0, -1):
        grid[y, 0] = p[idx]
        idx += 1

    # ---- 内部を補間 ----
    for y in range(10):
        for x in range(10):
            if x == 0 or x == 9 or y == 0 or y == 9:
                continue

            fx = x / 9.0
            fy = y / 9.0

            top = (1 - fx) * grid[0, 0] + fx * grid[0, 9]
            bottom = (1 - fx) * grid[9, 0] + fx * grid[9, 9]
            grid[y, x] = (1 - fy) * top + fy * bottom

    # ---- フル解像度へ拡大 ----
    map_x = cv2.resize(
        grid[..., 0],
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )

    map_y = cv2.resize(
        grid[..., 1],
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )

    return map_x, map_y


# ------------------------------------------------------------
# prepare_warp
# ------------------------------------------------------------
def prepare_warp(display_name, mode, src_size, log_func=None):
    width, height = src_size

    def _log(msg):
        if log_func:
            log_func(msg)
        else:
            print(msg)

    points = load_points(display_name, mode)

    if points is None:
        _log(f"[warp] grid NOT FOUND ({display_name}, {mode})")
        return None

    _log(f"[warp] building warp ({display_name}, {mode})")

    if mode == "perspective":
        map_x, map_y = _build_perspective_map(points, width, height)
    elif mode == "warp_map":
        map_x, map_y = _build_boundary_grid_map(points, width, height)
    else:
        raise RuntimeError(f"Unsupported warp mode: {mode}")


    return map_x.astype(np.float32), map_y.astype(np.float32)


# ------------------------------------------------------------
# GPU helper
# ------------------------------------------------------------
def convert_maps_to_uv_texture_data(map_x, map_y, width, height):
    u = map_x / float(width)
    v = map_y / float(height)
    uv = np.dstack((u, v)).astype(np.float32)
    return uv.tobytes()
