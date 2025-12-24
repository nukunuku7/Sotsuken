# precompute_warp_maps.py
import os
import time
import numpy as np
from pathlib import Path
from scipy.interpolate import Rbf

BASE_DIR = Path(__file__).resolve().parent
from config.environment_config import environment_config

WARP_CACHE_DIR = BASE_DIR / "config" / "warp_cache"
WARP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# Math helpers
# =========================================================
def normalize(v):
    v = np.asarray(v, np.float64)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def fit_plane(points):
    pts = np.asarray(points, np.float64)
    centroid = pts.mean(axis=0)
    cov = np.cov((pts - centroid).T)
    _, eigvecs = np.linalg.eigh(cov)
    u = normalize(eigvecs[:, 2])
    v = normalize(eigvecs[:, 1])
    return centroid, u, v

def build_ideal_rectangle_from_outline(dst_uv):
    """
    dst_uv : (N,2) 歪んだスクリーン外周 (0-1正規化)
    return : (N,2) 理想矩形外周
    """
    # 外周の累積距離
    diff = np.diff(np.vstack([dst_uv, dst_uv[0]]), axis=0)
    seglen = np.linalg.norm(diff, axis=1)
    s = np.insert(np.cumsum(seglen), 0, 0.0)
    s /= s[-1]

    src = np.zeros_like(dst_uv)

    for i, t in enumerate(s[:-1]):
        if t < 0.25:
            src[i] = [t * 4.0, 0.0]
        elif t < 0.50:
            src[i] = [1.0, (t - 0.25) * 4.0]
        elif t < 0.75:
            src[i] = [1.0 - (t - 0.50) * 4.0, 1.0]
        else:
            src[i] = [0.0, 1.0 - (t - 0.75) * 4.0]

    return src

# =========================================================
# Main
# =========================================================
def compute_all_maps(src_size):
    w, h = src_size

    for sim in environment_config["screen_simulation_sets"]:
        name = sim["name"]
        print(f"\n[compute] {name}  map size={w}x{h}")

        screen_pts = np.asarray(sim["screen"]["vertices"], np.float64)

        # -------------------------------------------------
        # 1. plane fitting
        # -------------------------------------------------
        sc_c, sc_u, sc_v = fit_plane(screen_pts)

        # -------------------------------------------------
        # 2. project 3D screen outline -> 2D
        # -------------------------------------------------
        uv = np.stack([
            np.dot(screen_pts - sc_c, sc_u),
            np.dot(screen_pts - sc_c, sc_v),
        ], axis=1)

        umin, vmin = uv.min(axis=0)
        umax, vmax = uv.max(axis=0)

        dst_uv = np.empty_like(uv)
        dst_uv[:, 0] = (uv[:, 0] - umin) / (umax - umin)
        dst_uv[:, 1] = (uv[:, 1] - vmin) / (vmax - vmin)

        # -------------------------------------------------
        # 3. build ideal rectangle (source side)
        # -------------------------------------------------
        src_uv = build_ideal_rectangle_from_outline(dst_uv)

        # -------------------------------------------------
        # 4. TPS : distorted(screen) -> ideal(source)
        # -------------------------------------------------
        print("[TPS] building thin plate spline...")
        tps_u = Rbf(dst_uv[:, 0], dst_uv[:, 1],
                    src_uv[:, 0], function="thin_plate")
        tps_v = Rbf(dst_uv[:, 0], dst_uv[:, 1],
                    src_uv[:, 1], function="thin_plate")

        # -------------------------------------------------
        # 5. build warp UV map
        # -------------------------------------------------
        map_uv = np.full((h, w, 2), -1.0, np.float32)

        start = time.time()

        for y in range(h):
            v = y / (h - 1)

            for x in range(w):
                u = x / (w - 1)

                uu = float(tps_u(u, v))
                vv = float(tps_v(u, v))

                if 0.0 <= uu <= 1.0 and 0.0 <= vv <= 1.0:
                    map_uv[y, x, 0] = uu
                    map_uv[y, x, 1] = vv

            progress = (y + 1) / h
            elapsed = time.time() - start
            eta = elapsed / progress - elapsed if progress > 0 else 0

            print(
                f"[PROGRESS] {y+1:4d}/{h} "
                f"({progress*100:5.1f}%) "
                f"elapsed={elapsed:6.1f}s "
                f"ETA={eta:6.1f}s"
            )

        path = WARP_CACHE_DIR / f"{name}_map_{w}x{h}.npz"

        np.savez(
            path,
            map_x=map_uv[..., 0].astype(np.float32),
            map_y=map_uv[..., 1].astype(np.float32),
        )

        print(f"[SAVE] {path}")

# =========================================================
if __name__ == "__main__":
    compute_all_maps((1920, 1080))
