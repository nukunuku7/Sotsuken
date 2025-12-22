import os
import math
import time
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

from config.environment_config import environment_config

WARP_CACHE_DIR = os.path.join("config", "warp_cache")
os.makedirs(WARP_CACHE_DIR, exist_ok=True)

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

def _fit_plane(points):
    pts = np.asarray(points, np.float64)
    centroid = pts.mean(axis=0)
    cov = np.cov((pts - centroid).T)
    _, eigvecs = np.linalg.eigh(cov)
    return centroid, _normalize(eigvecs[:,2]), _normalize(eigvecs[:,1])

def _nearest_along_ray(ray_o, ray_d, points):
    vecs = points - ray_o[None, :]
    t = np.dot(vecs, ray_d)
    mask = t > 0
    if not np.any(mask):
        return None
    cand = points[mask]
    proj = ray_o + t[mask][:,None] * ray_d
    idx = np.argmin(np.linalg.norm(proj - cand, axis=1))
    return cand[idx]

def _estimate_normals(pts, k=16):
    normals = np.zeros_like(pts)
    for i,p in enumerate(pts):
        d = np.linalg.norm(pts - p, axis=1)
        neigh = pts[np.argsort(d)[:k]]
        cov = np.cov((neigh - neigh.mean(0)).T)
        _, eigvecs = np.linalg.eigh(cov)
        normals[i] = _normalize(eigvecs[:,0])
    return normals

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

        uv = np.stack([
            np.dot(screen_pts - sc_c, sc_u),
            np.dot(screen_pts - sc_c, sc_v)
        ], axis=1)
        umin, vmin = uv.min(0)
        umax, vmax = uv.max(0)

        right = _normalize(np.cross(proj_d, [0, 0, 1]))
        up = _normalize(np.cross(right, proj_d))

        map_x = np.zeros((h, w), np.float32)
        map_y = np.zeros((h, w), np.float32)

        start = time.time()

        for y in range(h):
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
        )
        np.savez(path, map_x=map_x, map_y=map_y)
        print(f"  saved -> {path}")

if __name__ == "__main__":
    compute_all_maps((1920, 1080))
