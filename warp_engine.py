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

    w, h = src_size

    # ==================================================
    # load points
    # ==================================================
    pts = None
    if load_points_func:
        try:
            pts = load_points_func(display_name, mode)
        except Exception:
            pts = None

    # ==================================================
    # slice → full screen mapping
    # ==================================================
    slice_left = 0
    full_width = w

    key = display_name.replace("\\", "").replace(".", "")
    if key in DISPLAY_TO_SIMSET:
        info = DISPLAY_TO_SIMSET[key]
        slice_left = int(info.get("left", 0))
        full_width = int(info.get("full_width", w))

    # ==================================================
    # base grid（LOCAL 0–1）
    # ==================================================
    xs = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    ys = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))

    grid_u_local = xs / max(w - 1, 1)
    grid_v = ys / max(h - 1, 1)

    # ==================================================
    # ★ perspective は perspective モードのときだけ ★
    # ==================================================
    if mode == "perspective" and pts is not None and len(pts) >= 4:
        grid_pts = np.array(pts[:4], np.float32)

        if grid_pts.max() > 1.5:
            grid_pts[:, 0] *= w / 1920.0
            grid_pts[:, 1] *= h / 1080.0

        grid_matrix = generate_perspective_matrix(src_size, grid_pts)

        grid_u_local = cv2.warpPerspective(
            grid_u_local, grid_matrix, (w, h)
        )
        grid_v = cv2.warpPerspective(
            grid_v, grid_matrix, (w, h)
        )

        grid_u_local = np.clip(grid_u_local, 0.0, 1.0)
        grid_v = np.clip(grid_v, 0.0, 1.0)

    # ==================================================
    # LOCAL → GLOBAL 正規化
    # ==================================================
    grid_u = (slice_left + grid_u_local * (w - 1)) / max(full_width - 1, 1)

    # ==================================================
    # mask
    # ==================================================
    grid_mask = np.ones((h, w), dtype=bool)

    if mode == "warp_map" and pts is not None and len(pts) >= 8:
        poly = np.array(pts, np.float32)

        if poly.max() <= 1.5:
            poly[:, 0] *= w
            poly[:, 1] *= h
        else:
            poly[:, 0] *= w / 1920.0
            poly[:, 1] *= h / 1080.0

        poly_i = poly.astype(np.int32)
        mask_img = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask_img, [poly_i.reshape(-1, 1, 2)], 1)

        if np.count_nonzero(mask_img) > 0:
            grid_mask = mask_img.astype(bool)

    _log(f"[DEBUG] grid_mask valid pixels = {np.count_nonzero(grid_mask)}", log_func)

    # ==================================================
    # perspective モードはここで終了
    # ==================================================
    if mode == "perspective":
        warp_cache[cache_key] = (
            grid_u.astype(np.float32),
            grid_v.astype(np.float32)
        )
        return warp_cache[cache_key]

    # ==================================================
    # warp_map（以下は完全に元のまま）
    # ==================================================
    if environment_config is None:
        return None

    import re
    m = re.search(r"(\d+)$", display_name)
    if not m:
        return None

    display_num = int(m.group(1))
    sim_set_number = max(1, 5 - display_num)
    sim_idx = sim_set_number - 1

    sets = environment_config["screen_simulation_sets"]
    sim_idx = max(0, min(sim_idx, len(sets) - 1))
    sim_set = sets[sim_idx]

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
    screen_centroid, screen_u, screen_v, screen_n = _fit_plane(screen_pts)

    if np.dot(np.cross(screen_u, screen_v), screen_n) < 0:
        screen_v = -screen_v

    uv = np.stack([
        np.dot(screen_pts - screen_centroid, screen_u),
        np.dot(screen_pts - screen_centroid, screen_v)
    ], axis=1)

    umin, vmin = uv.min(axis=0)
    umax, vmax = uv.max(axis=0)
    du = max(umax - umin, 1e-6)
    dv = max(vmax - vmin, 1e-6)

    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)

    right = _normalize(np.cross(proj_dir, [0, 0, 1]))
    up = _normalize(np.cross(right, proj_dir))

    mirror_hit = screen_hit = uv_written = 0

    for y in range(h):
        for x in range(w):

            if not grid_mask[y, x]:
                continue

            gu = grid_u[y, x]
            gv = grid_v[y, x]

            nx = 2.0 * gu - 1.0
            ny = 2.0 * gv - 1.0

            sx = nx * math.tan(fov_h * 0.5)
            sy = ny * math.tan(fov_v * 0.5)

            ray = _normalize(proj_dir + right * sx + up * sy)

            hit, _, _ = _nearest_along_ray(proj_origin, ray, mirror_pts)
            if hit is None:
                continue
            mirror_hit += 1

            n = mirror_normals[np.argmin(np.linalg.norm(mirror_pts - hit, axis=1))]
            refl = _reflect(ray, n)

            sh, _, _ = _nearest_along_ray(hit + refl * 1e-6, refl, screen_pts)
            if sh is None:
                continue
            screen_hit += 1

            rel = sh - screen_centroid
            fu = (np.dot(rel, screen_u) - umin) / du
            fv = (np.dot(rel, screen_v) - vmin) / dv

            if 0.0 <= fu <= 1.0 and 0.0 <= fv <= 1.0:
                map_x[y, x] = fu
                map_y[y, x] = 1.0 - fv
                uv_written += 1

    _log(
        f"[DEBUG] mirror_hit={mirror_hit}, screen_hit={screen_hit}, uv_written={uv_written}",
        log_func
    )

    warp_cache[cache_key] = (
        map_x.astype(np.float32),
        map_y.astype(np.float32)
    )
    return warp_cache[cache_key]

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
