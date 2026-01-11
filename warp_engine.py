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
    # perspective
    # ==================================================
    if mode == "perspective" and pts is not None and len(pts) >= 4:
        grid_pts = np.array(pts[:4], np.float32)
        if grid_pts.max() > 1.5:
            grid_pts[:, 0] *= w / 1920.0
            grid_pts[:, 1] *= h / 1080.0

        M = generate_perspective_matrix(src_size, grid_pts)
        grid_u_local = cv2.warpPerspective(grid_u_local, M, (w, h))
        grid_v = cv2.warpPerspective(grid_v, M, (w, h))

        grid_u_local = np.clip(grid_u_local, 0.0, 1.0)
        grid_v = np.clip(grid_v, 0.0, 1.0)

    # ==================================================
    # LOCAL → GLOBAL
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

        mask_img = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask_img, [poly.astype(np.int32)], 1)
        if np.count_nonzero(mask_img) > 0:
            grid_mask = mask_img.astype(bool)

    if mode == "perspective":
        warp_cache[cache_key] = (
            grid_u.astype(np.float32),
            grid_v.astype(np.float32)
        )
        return warp_cache[cache_key]

    # ==================================================
    # warp_map : 自動ゆがみ補正（分離設計）
    # ==================================================
    if mode != "warp_map" or pts is None:
        return None

    grid = build_10x10_grid_from_points(pts, w, h, log_func)
    if grid is None:
        return None

    grid = normalize_grid_by_outer(grid)

    map_x, map_y = build_warp_map_from_grid(grid, w, h)

    warp_cache[cache_key] = (map_x, map_y)
    return warp_cache[cache_key]

# ----------------------------------------------------------------------
# Grid utilities
# ----------------------------------------------------------------------
def build_10x10_grid_from_points(pts, w, h, log_func=None):
    """
    pts: 36 or 100 points (pixel space)
    return: grid[10,10,2] in pixel space
    """
    pts = np.asarray(pts, np.float32)

    # ---- normalize input to pixel ----
    if pts.max() <= 1.5:
        pts[:, 0] *= (w - 1)
        pts[:, 1] *= (h - 1)
    else:
        pts[:, 0] *= w / 1920.0
        pts[:, 1] *= h / 1080.0

    # ---- full grid ----
    if len(pts) == 100:
        return pts.reshape(10, 10, 2)

    # ---- outer 36 only ----
    if len(pts) != 36:
        _log("[warp_map] pts must be 36 or 100", log_func)
        return None

    grid = np.zeros((10, 10, 2), np.float32)
    idx = 0

    grid[0, :]        = pts[idx:idx+10]; idx += 10
    grid[1:9, 9]      = pts[idx:idx+8];  idx += 8
    grid[9, ::-1]     = pts[idx:idx+10]; idx += 10
    grid[8:0:-1, 0]   = pts[idx:idx+8]

    # ---- interpolate interior ----
    for y in range(1, 9):
        for x in range(1, 9):
            grid[y, x] = (
                grid[y, 0] * (1 - x / 9.0) +
                grid[y, 9] * (x / 9.0)
            )

    return grid

# ----------------------------------------------------------------------
# build_ideal_grid
# ----------------------------------------------------------------------
def build_ideal_grid():
    ideal = np.zeros((10, 10, 2), np.float32)
    for j in range(10):
        for i in range(10):
            ideal[j, i] = [i / 9.0, j / 9.0]
    return ideal

# ----------------------------------------------------------------------
# normalize_grid_by_outer
# ----------------------------------------------------------------------
def normalize_grid_by_outer(grid):
    norm = grid.copy()

    for y in range(1, 9):
        ty = y / 9.0
        top    = grid[0, :]
        bottom = grid[9, :]

        for x in range(1, 9):
            tx = x / 9.0

            p_top    = top[x]
            p_bottom = bottom[x]
            p_left   = grid[y, 0]
            p_right  = grid[y, 9]

            # 横補間
            ph = p_left * (1 - tx) + p_right * tx
            # 縦補間
            pv = p_top * (1 - ty) + p_bottom * ty

            # 合成（平均）
            norm[y, x] = (ph + pv) * 0.5

    return norm

# ----------------------------------------------------------------------
# build_10x10_grid_from_points
# ----------------------------------------------------------------------
def build_warp_map_from_grid(grid, w, h):
    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)

    for y in range(h):
        v = y / (h - 1) * 9.0
        j = int(np.clip(math.floor(v), 0, 8))
        fv = v - j

        for x in range(w):
            u = x / (w - 1) * 9.0
            i = int(np.clip(math.floor(u), 0, 8))
            fu = u - i

            # 実スクリーン上の4点
            p00 = grid[j,     i]
            p10 = grid[j,     i + 1]
            p01 = grid[j + 1, i]
            p11 = grid[j + 1, i + 1]

            p = (
                p00 * (1 - fu) * (1 - fv) +
                p10 * fu       * (1 - fv) +
                p01 * (1 - fu) * fv +
                p11 * fu       * fv
            )

            # ★逆写像として設定
            map_x[y, x] = p[0]
            map_y[y, x] = p[1]

    # OpenCV 用に正規化
    map_x /= (w - 1)
    map_y /= (h - 1)

    return map_x, map_y

# ----------------------------------------------------------------------
# warp_image
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

    
