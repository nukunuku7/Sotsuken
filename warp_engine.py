# warp_engine.py
# ------------------------------------------------------------
# Warp Engine (Runtime Only / Safe Version)
#
# ・Blender による事前計算済み warp map (.npz) を「読むだけ」
# ・Runtime では一切の物理計算・crop・補間生成を行わない
# ・事前計算ファイルが無ければ、明示的にエラーを出して停止
# ------------------------------------------------------------

import os
import sys
import numpy as np
import cv2

# ----------------------------------------------------------------------
# Warp cache directory
# ----------------------------------------------------------------------
WARP_CACHE_DIR = os.path.join("config", "warp_cache")
os.makedirs(WARP_CACHE_DIR, exist_ok=True)

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

def _warp_cache_path(display_name, src_size):
    w, h = src_size
    return os.path.join(
        WARP_CACHE_DIR,
        f"{display_name}_map_{w}x{h}.npz"
    )

# ----------------------------------------------------------------------
# prepare_warp (Runtime Only)
# ----------------------------------------------------------------------
def prepare_warp(
    display_name,
    mode,
    src_size,
    load_points_func=None,  # 互換性のため残す（使用しない）
    log_func=None,
):
    """
    Runtime では mode="map" のみを許可する。
    それ以外は設計ミスとして即エラー。
    """

    # ==========================================================
    # mode チェック（事故防止）
    # ==========================================================
    if mode != "map":
        _log(
            "[warp] FATAL: runtime supports only mode='map'\n"
            f"        requested mode = '{mode}'\n"
            "        Please fix caller logic.",
            log_func,
        )
        raise RuntimeError("Invalid warp mode at runtime")

    display_name = _sanitize(display_name)
    cache_key = (display_name, src_size)

    # ==========================================================
    # ① メモリキャッシュ
    # ==========================================================
    if cache_key in warp_cache:
        return warp_cache[cache_key]

    # ==========================================================
    # ② ファイルキャッシュ（必須）
    # ==========================================================
    cache_path = _warp_cache_path(display_name, src_size)

    if not os.path.exists(cache_path):
        _log(
            "[warp] FATAL: precomputed warp map not found\n"
            f"        expected file:\n"
            f"        {cache_path}\n\n"
            "        Please run:\n"
            "          python precompute_warp_maps.py\n"
            "        before starting the media player.",
            log_func,
        )
        # 強制停止（半端に続行させない）
        raise FileNotFoundError(cache_path)

    # ==========================================================
    # ③ 読み込み
    # ==========================================================
    _log(f"[warp] load precomputed map: {cache_path}", log_func)

    data = np.load(cache_path)
    map_x = data["map_x"].astype(np.float32)
    map_y = data["map_y"].astype(np.float32)

    warp_cache[cache_key] = (map_x, map_y)
    return map_x, map_y

# ----------------------------------------------------------------------
# warp_image (CPU, optional utility)
# ----------------------------------------------------------------------
def warp_image(image, map_x, map_y):
    """
    CPU 用 warp（デバッグ・検証用）
    Runtime 表示では GPU 側 UV warp を使用する想定
    """
    return cv2.remap(
        image,
        map_x,
        map_y,
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
    u = map_x / float(width)
    v = map_y / float(height)
    uv = np.dstack((u, v)).astype(np.float32)
    return uv.tobytes()
