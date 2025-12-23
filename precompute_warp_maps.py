import os
import math
import time
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
from config.environment_config import environment_config

WARP_CACHE_DIR = os.path.join("config", "warp_cache")
os.makedirs(WARP_CACHE_DIR, exist_ok=True)

# =========================================================
# Optoma Keystone inverse model (to be estimated)
# =========================================================
KEYSTONE_AMOUNT_RANGE = np.linspace(0.10, 0.35, 11)
KEYSTONE_GAMMA_RANGE  = np.linspace(1.5, 2.3, 9)

MIN_COMPRESS = 0.35   # safety clamp


def optoma_keystone_inverse(nx, ny, amount, gamma):
    dy = abs(ny)
    compress = 1.0 - amount * (dy ** gamma)
    compress = max(compress, MIN_COMPRESS)
    return nx * compress, ny


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
    return centroid, normalize(eigvecs[:, 2]), normalize(eigvecs[:, 1])


def nearest_along_ray(ray_o, ray_d, points):
    vecs = points - ray_o[None, :]
    t = np.dot(vecs, ray_d)
    mask = t > 0
    if not np.any(mask):
        return None
    proj = ray_o + t[mask][:, None] * ray_d
    cand = points[mask]
    idx = np.argmin(np.linalg.norm(proj - cand, axis=1))
    return cand[idx]


def estimate_normals(pts, k=16):
    normals = np.zeros_like(pts)
    for i, p in enumerate(pts):
        d = np.linalg.norm(pts - p, axis=1)
        neigh = pts[np.argsort(d)[:k]]
        cov = np.cov((neigh - neigh.mean(0)).T)
        _, eigvecs = np.linalg.eigh(cov)
        normals[i] = normalize(eigvecs[:, 0])
    return normals


def reflect(d, n):
    return d - 2 * np.dot(d, n) * n


# =========================================================
# Error evaluation (screen straightness)
# =========================================================
def straightness_error(points):
    pts = np.asarray(points)
    if len(pts) < 3:
        return 1e6
    centroid = pts.mean(axis=0)
    cov = np.cov((pts - centroid).T)
    eigvals, _ = np.linalg.eigh(cov)
    return eigvals[0]   # smallest variance → line error


# =========================================================
# Keystone auto estimation
# =========================================================
def estimate_keystone(sim):
    print("[ESTIMATE] Keystone parameters")

    proj   = sim["projector"]
    mirror = sim["mirror"]
    screen = sim["screen"]

    proj_o = np.array(proj["origin"])
    proj_d = normalize(np.array(proj["direction"]))
    fov_h  = math.radians(proj["fov_h"])
    fov_v  = math.radians(proj["fov_v"])

    mirror_pts = np.array(mirror["vertices"])
    mirror_n   = estimate_normals(mirror_pts)

    screen_pts = np.array(screen["vertices"])
    sc_c, sc_u, sc_v = fit_plane(screen_pts)

    right = normalize(np.cross(proj_d, [0, 0, 1]))
    up    = normalize(np.cross(right, proj_d))

    best_err = 1e9
    best = None

    for amount in KEYSTONE_AMOUNT_RANGE:
        for gamma in KEYSTONE_GAMMA_RANGE:
            vertical_line = []
            horizontal_line = []

            for t in np.linspace(-1, 1, 21):
                # vertical center line
                nx, ny = 0.0, t
                nxk, nyk = optoma_keystone_inverse(nx, ny, amount, gamma)

                ray = normalize(
                    proj_d
                    + right * math.tan(nxk * fov_h / 2)
                    + up    * math.tan(nyk * fov_v / 2)
                )

                hit = nearest_along_ray(proj_o, ray, mirror_pts)
                if hit is None:
                    continue
                idx = np.argmin(np.linalg.norm(mirror_pts - hit, axis=1))
                refl = reflect(ray, mirror_n[idx])
                sh = nearest_along_ray(hit + refl * 1e-6, refl, screen_pts)
                if sh is not None:
                    vertical_line.append(sh)

                # horizontal center line
                nx, ny = t, 0.0
                nxk, nyk = optoma_keystone_inverse(nx, ny, amount, gamma)

                ray = normalize(
                    proj_d
                    + right * math.tan(nxk * fov_h / 2)
                    + up    * math.tan(nyk * fov_v / 2)
                )

                hit = nearest_along_ray(proj_o, ray, mirror_pts)
                if hit is None:
                    continue
                idx = np.argmin(np.linalg.norm(mirror_pts - hit, axis=1))
                refl = reflect(ray, mirror_n[idx])
                sh = nearest_along_ray(hit + refl * 1e-6, refl, screen_pts)
                if sh is not None:
                    horizontal_line.append(sh)

            err = straightness_error(vertical_line) + straightness_error(horizontal_line)

            print(f"  test amount={amount:.3f} gamma={gamma:.2f} err={err:.6f}")

            if err < best_err:
                best_err = err
                best = (amount, gamma)

    print(f"[ESTIMATE DONE] KEYSTONE_AMOUNT={best[0]:.4f}  GAMMA={best[1]:.3f}")
    return best


# =========================================================
# Main
# =========================================================
def compute_all_maps(src_size):
    w, h = src_size

    for sim in environment_config["screen_simulation_sets"]:
        name = sim.get("name", "display")
        print(f"[compute] {name} {w}x{h}")

        amount, gamma = estimate_keystone(sim)

        proj   = sim["projector"]
        mirror = sim["mirror"]
        screen = sim["screen"]

        proj_o = np.array(proj["origin"])
        proj_d = normalize(np.array(proj["direction"]))
        fov_h  = math.radians(proj["fov_h"])
        fov_v  = math.radians(proj["fov_v"])

        right = normalize(np.cross(proj_d, [0, 0, 1]))
        up    = normalize(np.cross(right, proj_d))

        print("[AXIS] projector origin :", proj_o)
        print("[AXIS] projector dir    :", proj_d)
        print("[AXIS] right            :", right)
        print("[AXIS] up               :", up)
        print(f"[AXIS] fov_h={proj['fov_h']} fov_v={proj['fov_v']}")

        mirror_pts = np.array(mirror["vertices"])
        mirror_n   = estimate_normals(mirror_pts)

        screen_pts = np.array(screen["vertices"])
        sc_c, sc_u, sc_v = fit_plane(screen_pts)

        uv = np.stack([
            np.dot(screen_pts - sc_c, sc_u),
            np.dot(screen_pts - sc_c, sc_v)
        ], axis=1)
        umin, vmin = uv.min(axis=0)
        umax, vmax = uv.max(axis=0)

        map_x = np.zeros((h, w), np.float32)
        map_y = np.zeros((h, w), np.float32)

        start_time = time.time()
        total_rays = h * w
        hit_count = 0

        for y in range(h):
            ny = (y / (h - 1)) * 2 - 1
            row_hits = 0

            for x in range(w):
                nx = (x / (w - 1)) * 2 - 1

                nxk, nyk = optoma_keystone_inverse(nx, ny, amount, gamma)

                ray = normalize(
                    proj_d
                    + right * math.tan(nxk * fov_h / 2)
                    + up    * math.tan(nyk * fov_v / 2)
                )

                hit = nearest_along_ray(proj_o, ray, mirror_pts)
                if hit is None:
                    continue

                idx = np.argmin(np.linalg.norm(mirror_pts - hit, axis=1))
                refl = reflect(ray, mirror_n[idx])
                sh = nearest_along_ray(hit + refl * 1e-6, refl, screen_pts)
                if sh is None:
                    continue

                rel = sh - sc_c
                fu = (np.dot(rel, sc_u) - umin) / (umax - umin)
                fv = (np.dot(rel, sc_v) - vmin) / (vmax - vmin)

                if 0 <= fu <= 1 and 0 <= fv <= 1:
                    map_x[y, x] = fu * (w - 1)
                    map_y[y, x] = (1 - fv) * (h - 1)
                    hit_count += 1
                    row_hits += 1

            # ---- progress log ----
            elapsed = time.time() - start_time
            progress = (y + 1) / h
            eta = elapsed / progress - elapsed if progress > 0 else 0

            print(
                f"[PROGRESS] {y+1:4d}/{h} "
                f"({progress*100:5.1f}%) "
                f"row_hit={row_hits/w*100:5.1f}% "
                f"total_hit={hit_count/(w*(y+1))*100:5.1f}% "
                f"elapsed={elapsed:6.1f}s "
                f"ETA={eta:6.1f}s"
            )

        path = os.path.join(WARP_CACHE_DIR, f"{name}_map_{w}x{h}.npz")
        np.savez(path, map_x=map_x, map_y=map_y)

        print(f"[SAVE] warp map saved -> {path}")
        print(f"[SUMMARY] hit ratio = {hit_count/total_rays*100:.2f}%")

if __name__ == "__main__":
    compute_all_maps((1920, 1080))
