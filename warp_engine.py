# warp_engine.py (CPU専用版)
import os
import json
import math
import numpy as np
import cv2

# try import environment_config generated from Blender
try:
    from config.environment_config import environment_config
except Exception:
    environment_config = None

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
# Math helpers (CPU)
# ----------------------------------------------------------------------
def _normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def _fit_plane(points):
    pts = np.asarray(points, dtype=np.float64)
    centroid = pts.mean(axis=0)
    cov = np.cov((pts - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]
    u = eigvecs[:, 2]
    v = eigvecs[:, 1]
    return centroid, _normalize(u), _normalize(v), _normalize(normal)

def _nearest_along_ray(ray_o, ray_d, points, t_min=0.0, t_max=10.0):
    pts = np.asarray(points, dtype=np.float64)
    vecs = pts - ray_o[None, :]
    t_vals = np.dot(vecs, ray_d)
    mask = t_vals > t_min
    if not np.any(mask):
        return None, None, None
    candidate_pts = pts[mask]
    ts = t_vals[mask]
    projected = ray_o[None, :] + np.outer(ts, ray_d)
    dists = np.linalg.norm(projected - candidate_pts, axis=1)
    idx = np.argmin(dists)
    best_t = ts[idx]
    if best_t < t_min or best_t > t_max:
        return None, None, None
    return candidate_pts[idx], float(best_t), float(dists[idx])

def _estimate_normals_for_pointcloud(pts, k=16):
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    normals = np.zeros_like(pts)
    for i in range(n):
        dists = np.linalg.norm(pts - pts[i], axis=1)
        idx = np.argsort(dists)[:min(k, n)]
        neigh = pts[idx]
        centroid = neigh.mean(axis=0)
        cov = np.cov((neigh - centroid).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normals[i] = _normalize(eigvecs[:, 0])
    return normals

def _reflect(d, n):
    d = _normalize(d)
    n = _normalize(n)
    return d - 2.0 * np.dot(d, n) * n

# ----------------------------------------------------------------------
# Perspective matrix
# ----------------------------------------------------------------------
def generate_perspective_matrix(src_size, dst_points):
    w, h = src_size
    src_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    dst_pts = np.array(dst_points, np.float32)
    return cv2.getPerspectiveTransform(src_pts, dst_pts)

# ----------------------------------------------------------------------
# Display map
# ----------------------------------------------------------------------
DISPLAY_TO_SIMSET = {}
try:
    with open("config/display_map.json", "r", encoding="utf-8") as f:
        DISPLAY_TO_SIMSET = json.load(f).get("display_map", {})
except Exception:
    DISPLAY_TO_SIMSET = {}

# ----------------------------------------------------------------------
# prepare_warp (CPU計算のみ)
# ----------------------------------------------------------------------
def prepare_warp(display_name, mode, src_size,
                 overlap_px=0,
                 load_points_func=None,
                 log_func=None):

    display_name = (
        display_name.replace("(", "_")
                    .replace(")", "_")
                    .replace(" ", "_")
                    .replace("-", "_")
                    .replace("/", "_")
    )

    cache_key = (display_name, mode, src_size)
    if cache_key in warp_cache:
        return warp_cache[cache_key]

    # ---------------- perspective ----------------
    if mode == "perspective":
        if load_points_func:
            pts = load_points_func(display_name, mode)
        else:
            cfg = f"config/projector_profiles/__._{display_name}_{mode}_points.json"
            if not os.path.exists(cfg):
                return None
            with open(cfg, "r", encoding="utf-8") as f:
                pts = json.load(f)

        pts = np.array(pts[:4], np.float32)

        # ★ 先に定義する（重要）
        w, h = src_size

        # pts がフルHD基準だった場合を補正
        if pts.max() > w + 10:
            scale_x = w / 1920.0
            scale_y = h / 1080.0
            pts[:, 0] *= scale_x
            pts[:, 1] *= scale_y


        matrix = generate_perspective_matrix(src_size, pts)

        # ★ 正しいグリッド生成（X/Y を明示）
        # xs, ys は「キャプチャ画像全体」
        xs = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        ys = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))

        map_x = cv2.warpPerspective(xs, matrix, (w, h)) / (w - 1)
        map_y = cv2.warpPerspective(ys, matrix, (w, h)) / (h - 1)

        map_x = np.clip(map_x, 0.0, 1.0).astype(np.float32)
        map_y = np.clip(map_y, 0.0, 1.0).astype(np.float32)

        warp_cache[cache_key] = (map_x, map_y)

        _log(
            f"[WARP DEBUG] map_x min/max={map_x.min():.3f}/{map_x.max():.3f}, "
            f"map_y min/max={map_y.min():.3f}/{map_y.max():.3f}",
            log_func
        )

        return map_x, map_y

    # ---------------- warp_map (CPU heavy) ----------------
    if environment_config is None:
        return None

    sim_idx = DISPLAY_TO_SIMSET.get(display_name)
    if sim_idx is None:
        return None

    sim_set = environment_config["screen_simulation_sets"][sim_idx]
    proj = sim_set["projector"]
    mirror = sim_set["mirror"]
    screen = sim_set["screen"]

    proj_origin = np.array(proj["origin"], np.float64)
    proj_dir = _normalize(np.array(proj["direction"], np.float64))
    fov_h = math.radians(proj.get("fov_h", 53.13))
    fov_v = math.radians(proj.get("fov_v", fov_h))

    mirror_pts = np.array(mirror["vertices"], np.float64)
    screen_pts = np.array(screen["vertices"], np.float64)

    mirror_normals = _estimate_normals_for_pointcloud(mirror_pts)
    screen_centroid, screen_u, screen_v, screen_normal = _fit_plane(screen_pts)

    uv = np.stack([
        np.dot(screen_pts - screen_centroid, screen_u),
        np.dot(screen_pts - screen_centroid, screen_v)
    ], axis=1)

    umin, vmin = uv.min(axis=0)
    umax, vmax = uv.max(axis=0)

    w, h = src_size
    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)

    right = _normalize(np.cross(proj_dir, [0, 0, 1]))
    up = _normalize(np.cross(right, proj_dir))

    for y in range(h):
        for x in range(w):
            u = (x / w - 0.5) * fov_h
            v = (y / h - 0.5) * fov_v
            ray = _normalize(proj_dir + right * math.tan(u) + up * math.tan(v))

            hit, _, _ = _nearest_along_ray(proj_origin, ray, mirror_pts)
            if hit is None:
                continue

            n = mirror_normals[np.argmin(np.linalg.norm(mirror_pts - hit, axis=1))]
            refl = _reflect(ray, n)

            sh, _, _ = _nearest_along_ray(hit + refl * 1e-6, refl, screen_pts)
            if sh is None:
                continue

            rel = sh - screen_centroid
            fu = (np.dot(rel, screen_u) - umin) / (umax - umin)
            fv = (np.dot(rel, screen_v) - vmin) / (vmax - vmin)

            if 0 <= fu <= 1 and 0 <= fv <= 1:
                map_x[y, x] = fu * (w - 1)
                map_y[y, x] = (1 - fv) * (h - 1)

    warp_cache[cache_key] = (map_x, map_y)
    return map_x, map_y

# ----------------------------------------------------------------------
# warp_image (CPUのみ)
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
