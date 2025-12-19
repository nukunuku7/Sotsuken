# warp_engine.py
# ------------------------------------------------------------
# Warp Engine (CPU) - Runtime Only
# ・Blender事前計算済みwarp mapを読むだけ
# ・実行時に物理計算は一切行わない
# ------------------------------------------------------------

import os
import numpy as np
import cv2

# ----------------------------------------------------------------------
# Warp cache directory
# ----------------------------------------------------------------------
WARP_CACHE_DIR = os.path.join("config", "warp_cache")
os.makedirs(WARP_CACHE_DIR, exist_ok=True)

def _warp_cache_path(display_name, mode, src_size):
    """
    mode:
      - map        : ScreenSimulatorSet_X_map_WxH.npz
      - perspective / warp_map : 派生キャッシュ
    """
    w, h = src_size
    return os.path.join(
        WARP_CACHE_DIR,
        f"{display_name}_{mode}_{w}x{h}.npz"
    )

# ----------------------------------------------------------------------
# In-memory cache
# ----------------------------------------------------------------------
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
# Utility
# ----------------------------------------------------------------------
def _sanitize(name: str) -> str:
    return (
        name.replace("(", "_")
            .replace(")", "_")
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
    )

# ----------------------------------------------------------------------
# UV crop (map -> sub map)
# ----------------------------------------------------------------------
def _crop_uv(map_x, map_y, u0, v0, u1, v1):
    h, w = map_x.shape
    out_x = np.zeros_like(map_x)
    out_y = np.zeros_like(map_y)

    for y in range(h):
        fv = y / (h - 1)
        if not (v0 <= fv <= v1):
            continue

        fv_l = (fv - v0) / (v1 - v0)
        sy = int(fv_l * (h - 1))

        for x in range(w):
            fu = x / (w - 1)
            if not (u0 <= fu <= u1):
                continue

            fu_l = (fu - u0) / (u1 - u0)
            sx = int(fu_l * (w - 1))

            out_x[y, x] = map_x[sy, sx]
            out_y[y, x] = map_y[sy, sx]

    return out_x, out_y

# ----------------------------------------------------------------------
# prepare_warp
# ----------------------------------------------------------------------
def prepare_warp(display_name, mode, src_size, load_points_func=None, log_func=None):
    """
    mode:
      - "map"          : 事前計算済み物理warpを読むだけ
      - "perspective"  : mapからcrop
      - "warp_map"     : mapからcrop
    """

    display_name = _sanitize(display_name)
    cache_key = (display_name, mode, src_size)

    # ==========================================================
    # ① メモリキャッシュ
    # ==========================================================
    if cache_key in warp_cache:
        return warp_cache[cache_key]

    # ==========================================================
    # ② ファイルキャッシュ
    # ==========================================================
    cache_path = _warp_cache_path(display_name, mode, src_size)
    if os.path.exists(cache_path):
        _log(f"[warp] load cache: {cache_path}", log_func)
        data = np.load(cache_path)
        map_x = data["map_x"]
        map_y = data["map_y"]
        warp_cache[cache_key] = (map_x, map_y)
        return map_x, map_y

    # ==========================================================
    # map は「読むだけ」
    # ==========================================================
    if mode == "map":
        _log(
            f"[warp] ERROR: precomputed map not found:\n  {cache_path}",
            log_func
        )
        return None

    # ==========================================================
    # perspective / warp_map (mapからcrop)
    # ==========================================================
    _log(f"[warp] build {mode} from map: {display_name}", log_func)

    base = prepare_warp(display_name, "map", src_size, load_points_func, log_func)
    if base is None:
        return None

    if load_points_func is None:
        _log("[warp] ERROR: load_points_func is None", log_func)
        return None

    base_x, base_y = base
    pts = np.asarray(load_points_func(display_name, mode), np.float32)

    if len(pts) == 0:
        _log("[warp] ERROR: no control points", log_func)
        return None

    w, h = src_size
    us = pts[:, 0] / (w - 1)
    vs = pts[:, 1] / (h - 1)

    u0, u1 = us.min(), us.max()
    v0, v1 = vs.min(), vs.max()

    _log(
        f"[warp] crop area u=({u0:.3f},{u1:.3f}) "
        f"v=({v0:.3f},{v1:.3f})",
        log_func
    )

    map_x, map_y = _crop_uv(base_x, base_y, u0, v0, u1, v1)

    np.savez(
        cache_path,
        map_x=map_x.astype(np.float32),
        map_y=map_y.astype(np.float32),
    )

    warp_cache[cache_key] = (map_x, map_y)
    _log(f"[warp] cache saved: {cache_path}", log_func)

    return map_x, map_y

# ----------------------------------------------------------------------
# warp_image (CPU)
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

# ----------------------------------------------------------------------
# UV map for ModernGL (pixel -> normalized UV)
# ----------------------------------------------------------------------
def convert_maps_to_uv_texture_data(map_x, map_y, width, height):
    """
    map_x, map_y : pixel coordinate maps (float32)
    width, height: source image size
    return        : bytes for GL_RG32F texture (u, v in 0–1)
    """
    u = map_x.astype(np.float32) / float(width)
    v = map_y.astype(np.float32) / float(height)
    uv = np.dstack((u, v))
    return uv.astype("f4").tobytes()
