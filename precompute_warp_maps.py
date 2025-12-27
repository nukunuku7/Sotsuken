# ================================
# precompute_warp_maps.py (REVISED)
# ================================
# 役割:
# - environment_config.py に保存された
#   スクリーン点群のみを使用して補正マップ(map_x,map_y)を生成
# - 実行時(media_player)では一切再計算しない

import time
import numpy as np
from pathlib import Path
from scipy.interpolate import Rbf
from config.environment_config import environment_config

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "config" / "warp_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fit_plane(points):
    pts = np.asarray(points, np.float64)
    c = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - c)
    u, v = vh[0], vh[1]
    return c, u / np.linalg.norm(u), v / np.linalg.norm(v)


def compute_all_maps(size):
    w, h = size

    for sim in environment_config["screen_simulation_sets"]:
        name = sim["name"]
        print(f"\n[START] Simulator: {name}")

        t0 = time.time()
        screen_pts = np.asarray(sim["screen"]["vertices"], np.float64)

        # ---- 平面当てはめ（スクリーン基準）
        c, u, v = fit_plane(screen_pts)
        uv = np.stack([
            np.dot(screen_pts - c, u),
            np.dot(screen_pts - c, v)
        ], axis=1)

        # ---- 0-1 正規化
        uv -= uv.min(axis=0)
        uv /= uv.max(axis=0)

        # ---- TPS : スクリーン歪み → 正方形
        tps_u = Rbf(uv[:, 0], uv[:, 1], uv[:, 0], function="thin_plate")
        tps_v = Rbf(uv[:, 0], uv[:, 1], uv[:, 1], function="thin_plate")

        map_x = np.zeros((h, w), np.float32)
        map_y = np.zeros((h, w), np.float32)

        for y in range(h):
            vv = y / (h - 1)
            for x in range(w):
                uu = x / (w - 1)
                map_x[y, x] = tps_u(uu, vv)
                map_y[y, x] = tps_v(uu, vv)

        np.savez(
            CACHE_DIR / f"{name}_map_{w}x{h}.npz",
            map_x=map_x,
            map_y=map_y
        )

        print(f"[SAVE] {name}_map_{w}x{h}.npz ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    compute_all_maps((1920, 1080))
