# warp_engine.py
# ------------------------------------------------------------
# Warp Engine (File-existence check only)
#
# ・grid_editor_* が保存した JSON の「存在」だけ確認
# ・perspective / warp_map を判別
# ・中身は読まない（座標計算は従来ロジックを使用）
# ------------------------------------------------------------

import os
import numpy as np
import cv2
from editor.grid_utils import log

PROJECTOR_PROFILE_DIR = os.path.join(
    "config", "projector_profiles"
)

# ------------------------------------------------------------
# File existence check
# ------------------------------------------------------------
def _grid_file_exists(display_id, mode):
    """
    display_id : "D2" など
    mode       : "perspective" or "map"
    """
    if mode == "perspective":
        filename = f"{display_id}_perspective_points.json"
    elif mode == "map":
        filename = f"{display_id}_warp_map_points.json"
    else:
        return False

    return os.path.exists(
        os.path.join(PROJECTOR_PROFILE_DIR, filename)
    )

# ------------------------------------------------------------
# Dummy warp builders (same as before)
# ------------------------------------------------------------
def _build_perspective_map(width, height):
    map_x, map_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    return map_x, map_y


def _build_grid_map(width, height):
    map_x, map_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    return map_x, map_y

# ------------------------------------------------------------
# prepare_warp
# ------------------------------------------------------------
def prepare_warp(
    display_name,
    mode,
    src_size,
    log_func=None,
):
    """
    display_name : "D2" のようなID
    mode         : "perspective" or "map"
    src_size     : (width, height)
    """

    width, height = src_size

    if not _grid_file_exists(display_name, mode):
        log(
            f"[warp] grid file NOT FOUND "
            f"(display={display_name}, mode={mode})",
            log_func,
        )
        return None

    log(
        f"[warp] grid file FOUND "
        f"(display={display_name}, mode={mode})",
        log_func,
    )

    # 実際の歪みは「前の安定していた方法」を使う前提
    if mode == "perspective":
        map_x, map_y = _build_perspective_map(width, height)
    elif mode == "map":
        map_x, map_y = _build_grid_map(width, height)
    else:
        raise RuntimeError(f"Unsupported warp mode: {mode}")

    return map_x, map_y

# ------------------------------------------------------------
# GPU helper
# ------------------------------------------------------------
def convert_maps_to_uv_texture_data(map_x, map_y, width, height):
    u = map_x / float(width)
    v = map_y / float(height)
    uv = np.dstack((u, v)).astype(np.float32)
    return uv.tobytes()
