# precompute_warp_maps.py (FINAL + progress debug)

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


def build_rectangle(dst):
    diff = np.diff(np.vstack([dst, dst[0]]), axis=0)
    s = np.insert(np.cumsum(np.linalg.norm(diff, axis=1)), 0, 0)
    s /= s[-1]

    src = np.zeros_like(dst)
    for i, t in enumerate(s[:-1]):
        if t < 0.25:
            src[i] = [t * 4, 0]
        elif t < 0.5:
            src[i] = [1, (t - 0.25) * 4]
        elif t < 0.75:
            src[i] = [1 - (t - 0.5) * 4, 1]
        else:
            src[i] = [0, 1 - (t - 0.75) * 4]
    return src


def compute_all_maps(size):
    w, h = size

    for sim in environment_config["screen_simulation_sets"]:
        name = sim["name"]
        print(f"\n[START] Simulator: {name}")

        t0 = time.time()
        pts = np.asarray(sim["screen"]["vertices"], np.float64)

        # ---- Plane fitting
        c, u, v = fit_plane(pts)
        uv = np.stack([np.dot(pts - c, u), np.dot(pts - c, v)], axis=1)

        uv -= uv.min(axis=0)
        uv /= uv.max(axis=0)

        print("  - Build TPS interpolator ...")
        src = build_rectangle(uv)

        tps_u = Rbf(uv[:, 0], uv[:, 1], src[:, 0], function="thin_plate")
        tps_v = Rbf(uv[:, 0], uv[:, 1], src[:, 1], function="thin_plate")

        print(f"  - Allocating warp maps: {w}x{h}")
        map_x = np.full((h, w), -1, np.float32)
        map_y = np.full((h, w), -1, np.float32)

        print("  - Computing warp map rows...")
        row_start = time.time()
        last_print = 0

        for y in range(h):
            v0 = y / (h - 1)

            for x in range(w):
                u0 = x / (w - 1)
                uu = float(tps_u(u0, v0))
                vv = float(tps_v(u0, v0))
                if 0 <= uu <= 1 and 0 <= vv <= 1:
                    map_x[y, x] = uu
                    map_y[y, x] = vv

            # ---- progress output every ~10%
            progress = y / (h - 1)
            if progress - last_print >= 0.1 or y == 0 or y == h - 1:
                elapsed = time.time() - row_start
                eta = elapsed / max(progress, 1e-6) * (1.0 - progress)
                print(
                    f"    row {y:4d} / {h} "
                    f"({progress*100:5.1f}%) "
                    f"elapsed {elapsed:6.1f}s, ETA {eta:6.1f}s"
                )
                last_print = progress

        total = time.time() - t0
        print(f"[DONE] Compute warp map: {total:.1f}s")

        np.savez(
            CACHE_DIR / f"{name}_map_{w}x{h}.npz",
            map_x=map_x,
            map_y=map_y
        )
        print(f"[SAVE] {name}_map_{w}x{h}.npz")


if __name__ == "__main__":
    compute_all_maps((1920, 1080))
