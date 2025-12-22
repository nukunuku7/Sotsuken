<<<<<<< HEAD
# ================================
# precompute_warp_maps.py (REVISED)
# ================================
# 役割:
# - environment_config.py に保存された
#   スクリーン点群のみを使用して補正マップ(map_x,map_y)を生成
# - 実行時(media_player)では一切再計算しない

=======
import os
import math
>>>>>>> 97c68d26 (システム的には完全なる完成をしました。)
import time
import numpy as np
from pathlib import Path
from scipy.interpolate import Rbf
from config.environment_config import environment_config

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "config" / "warp_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

<<<<<<< HEAD
=======
# ===========================
# Optoma internal DSP model
# ===========================
KEYSTONE_V_DEG = 40.0          # Projector setting (+40)
KEYSTONE_STRENGTH = math.tan(math.radians(KEYSTONE_V_DEG))
KEYSTONE_GAMMA = 1.85          # Optoma-like nonlinearity

# ---------------- math helpers ----------------
def _normalize(v):
    v = np.asarray(v, np.float64)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n
>>>>>>> 97c68d26 (システム的には完全なる完成をしました。)

def fit_plane(points):
    pts = np.asarray(points, np.float64)
    c = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - c)
    u, v = vh[0], vh[1]
    return c, u / np.linalg.norm(u), v / np.linalg.norm(v)


def compute_all_maps(size):
    w, h = size

<<<<<<< HEAD
    for sim in environment_config["screen_simulation_sets"]:
        name = sim["name"]
        print(f"\n[START] Simulator: {name}")

        t0 = time.time()
        screen_pts = np.asarray(sim["screen"]["vertices"], np.float64)
=======
def _reflect(d, n):
    return d - 2*np.dot(d, n)*n

# ---------------- main ----------------
def compute_all_maps(src_size):
    w, h = src_size

    sim_sets = environment_config["screen_simulation_sets"]

    for sim_idx, sim in enumerate(sim_sets):
        display_name = sim.get("name", f"display_{sim_idx}")
        print(f"[compute] {display_name} {w}x{h}")

        proj = sim["projector"]
        mirror = sim["mirror"]
        screen = sim["screen"]

        proj_o = np.array(proj["origin"])
        proj_d = _normalize(np.array(proj["direction"]))
        fov_h = math.radians(proj["fov_h"])
        fov_v = math.radians(proj["fov_v"])

        mirror_pts = np.array(mirror["vertices"])
        mirror_n = _estimate_normals(mirror_pts)

        screen_pts = np.array(screen["vertices"])
        sc_c, sc_u, sc_v = _fit_plane(screen_pts)
>>>>>>> 97c68d26 (システム的には完全なる完成をしました。)

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
<<<<<<< HEAD
            vv = y / (h - 1)
            for x in range(w):
                uu = x / (w - 1)
                map_x[y, x] = tps_u(uu, vv)
                map_y[y, x] = tps_v(uu, vv)

        np.savez(
            CACHE_DIR / f"{name}_map_{w}x{h}.npz",
            map_x=map_x,
            map_y=map_y
=======
            if y % max(1, h // 20) == 0:
                elapsed = time.time() - start
                print(f"  {int(y / h * 100)}%  elapsed: {elapsed:.1f}s")

            v_angle = (y / h - 0.5) * fov_v

            for x in range(w):
                u_angle = (x / w - 0.5) * fov_h

                ray = _normalize(
                    proj_d
                    + right * math.tan(u_angle)
                    + up * math.tan(v_angle)
                )

                hit = _nearest_along_ray(proj_o, ray, mirror_pts)
                if hit is None:
                    continue

                n = mirror_n[np.argmin(np.linalg.norm(mirror_pts - hit, axis=1))]
                refl = _reflect(ray, n)

                sh = _nearest_along_ray(hit + refl * 1e-6, refl, screen_pts)
                if sh is None:
                    continue

                rel = sh - sc_c
                fu = (np.dot(rel, sc_u) - umin) / (umax - umin)
                fv = (np.dot(rel, sc_v) - vmin) / (vmax - vmin)

                # ===== Optoma nonlinear keystone (image space) =====
                dy = fv - 0.5
                scale = 1.0 - KEYSTONE_STRENGTH * np.sign(dy) * (abs(dy) ** KEYSTONE_GAMMA)
                fu = (fu - 0.5) * scale + 0.5
                # ==================================================

                if 0 <= fu <= 1 and 0 <= fv <= 1:
                    map_x[y, x] = fu * (w - 1)
                    map_y[y, x] = (1 - fv) * (h - 1)

        path = os.path.join(
            WARP_CACHE_DIR,
            f"{display_name}_map_{w}x{h}.npz"
>>>>>>> 97c68d26 (システム的には完全なる完成をしました。)
        )

        print(f"[SAVE] {name}_map_{w}x{h}.npz ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    compute_all_maps((1920, 1080))
