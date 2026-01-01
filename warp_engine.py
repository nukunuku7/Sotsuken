# warp_engine.py
# ------------------------------------------------------------
# Warp Engine (Grid-based / Runtime Safe Version)
#
# ・grid_editor_* で保存された制御点(JSON)を読む
# ・perspective / warp_map の2方式をサポート
# ・precompute / npz / 物理計算は一切行わない
# ------------------------------------------------------------

import numpy as np
import cv2
from editor.grid_utils import load_points, log

# ------------------------------------------------------------
# Perspective warp
# ------------------------------------------------------------
def _build_perspective_map(points, width, height):
    """
    points: list of 4 points (normalized or pixel)
    """
    if len(points) != 4:
        raise ValueError("Perspective mode requires exactly 4 points")

    src = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height],
    ], dtype=np.float32)

    dst = np.array(points, dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)

    map_x, map_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )

    coords = np.stack([map_x, map_y, np.ones_like(map_x)], axis=-1)
    warped = coords @ M.T
    warped /= warped[..., 2:3]

    return warped[..., 0], warped[..., 1]

# ------------------------------------------------------------
# Grid warp
# ------------------------------------------------------------
def _build_grid_map(points, width, height, grid_size=6):
    """
    points: list of grid points (row-major)
    """
    if len(points) != grid_size * grid_size:
        raise ValueError("Warp map grid size mismatch")

    points = np.array(points, dtype=np.float32).reshape(
        (grid_size, grid_size, 2)
    )

    src_x = np.linspace(0, width, grid_size, dtype=np.float32)
    src_y = np.linspace(0, height, grid_size, dtype=np.float32)
    src = np.stack(np.meshgrid(src_x, src_y), axis=-1)

    map_x = cv2.resize(
        points[..., 0],
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )
    map_y = cv2.resize(
        points[..., 1],
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )

    return map_x, map_y

# ------------------------------------------------------------
# prepare_warp (main entry)
# ------------------------------------------------------------
def prepare_warp(
    display_name,
    mode,
    src_size,
    log_func=None,
):
    """
    Returns:
        map_x, map_y (float32 pixel coordinate maps)
    """

    width, height = src_size
    points = load_points(display_name, mode)

    if points is None:
        log(f"[warp] no grid data for {display_name}", log_func)
        return None

    log(f"[warp] building warp ({mode}) for {display_name}", log_func)

    if mode == "perspective":
        map_x, map_y = _build_perspective_map(points, width, height)

    elif mode == "map":
        map_x, map_y = _build_grid_map(points, width, height)

    else:
        raise RuntimeError(f"Unsupported warp mode: {mode}")

    return map_x.astype(np.float32), map_y.astype(np.float32)

# ------------------------------------------------------------
# GPU helper
# ------------------------------------------------------------
def convert_maps_to_uv_texture_data(map_x, map_y, width, height):
    """
    Pixel map → normalized UV (0–1)
    """
    u = map_x / float(width)
    v = map_y / float(height)
    uv = np.dstack((u, v)).astype(np.float32)
    return uv.tobytes()
