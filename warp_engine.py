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
    # warp_map : 曲線外周 → 曲線補間グリッド → セル単位射影押し込み
    # ==================================================
    if mode != "warp_map" or pts is None:
        return None

    pts_np = np.asarray(pts, np.float32)

    # ------------------------------
    # 正規化（pts → pixel）
    # ------------------------------
    if pts_np.max() <= 1.5:
        pts_np[:, 0] *= (w - 1)
        pts_np[:, 1] *= (h - 1)
    else:
        pts_np[:, 0] *= w / 1920.0
        pts_np[:, 1] *= h / 1080.0

    if len(pts_np) != 36:
        _log("[warp_map] require exactly 36 outer points", log_func)
        return None

    # ------------------------------
    # 外周分解（時計回り）
    # ------------------------------
    idx = 0
    top    = pts_np[idx:idx+10]; idx += 10
    right  = pts_np[idx:idx+8];  idx += 8
    bottom = pts_np[idx:idx+10]; idx += 10
    left   = pts_np[idx:idx+8]

    bottom = bottom[::-1]
    left   = left[::-1]

    # ------------------------------
    # 距離ベース等間隔再サンプリング
    # ------------------------------
    def resample_curve(curve, n):
        d = np.linalg.norm(curve[1:] - curve[:-1], axis=1)
        s = np.concatenate([[0.0], np.cumsum(d)])
        t = np.linspace(0, s[-1], n)
        x = np.interp(t, s, curve[:, 0])
        y = np.interp(t, s, curve[:, 1])
        return np.stack([x, y], axis=1)

    top    = resample_curve(top,    10)
    right  = resample_curve(right,  10)
    bottom = resample_curve(bottom, 10)
    left   = resample_curve(left,   10)

    # ------------------------------
    # Coons Patch による 10x10 グリッド生成
    # ------------------------------
    grid = np.zeros((10, 10, 2), np.float32)

    for j in range(10):
        v = j / 9.0
        for i in range(10):
            u = i / 9.0

            B = (
                (1 - v) * top[i] +
                v * bottom[i] +
                (1 - u) * left[j] +
                u * right[j]
            )

            C = (
                (1 - u) * (1 - v) * top[0] +
                u * (1 - v) * top[9] +
                (1 - u) * v * bottom[0] +
                u * v * bottom[9]
            )

            grid[j, i] = B - C

    # ------------------------------
    # 各セルの逆射影行列＋マスク生成
    # ------------------------------
    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)

    filled_mask = np.zeros((h, w), np.uint8)

    for j in range(9):
        for i in range(9):
            dst = np.array([
                grid[j,   i],
                grid[j,   i+1],
                grid[j+1, i+1],
                grid[j+1, i]
            ], np.float32)

            src = np.array([
                [i / 9,     j / 9],
                [(i+1)/9,   j / 9],
                [(i+1)/9, (j+1)/9],
                [i / 9,   (j+1)/9]
            ], np.float32)

            Hinv = cv2.getPerspectiveTransform(dst, src)

            # セルマスク作成
            cell_mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(cell_mask, [dst.astype(np.int32)], 1)

            ys, xs = np.where((cell_mask == 1) & (filled_mask == 0))
            if len(xs) == 0:
                continue

            pts_xy = np.stack([xs, ys, np.ones_like(xs)], axis=1).astype(np.float32)
            uvw = (Hinv @ pts_xy.T).T

            valid = uvw[:, 2] != 0.0
            uv = np.zeros((len(xs), 2), np.float32)
            uv[valid, 0] = uvw[valid, 0] / uvw[valid, 2]
            uv[valid, 1] = uvw[valid, 1] / uvw[valid, 2]

            map_x[ys, xs] = np.clip(uv[:, 0], 0.0, 1.0)
            map_y[ys, xs] = np.clip(uv[:, 1], 0.0, 1.0)
            filled_mask[ys, xs] = 1

    warp_cache[cache_key] = (map_x, map_y)
    return warp_cache[cache_key]

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
