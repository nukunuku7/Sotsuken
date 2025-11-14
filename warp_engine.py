#!/usr/bin/env python3
# warp_engine.py
# Convex Mirror + 360° Screen RayTracing Warp Engine

import os
import numpy as np
import cv2
from math import tan, radians

warp_cache = {}

# ============================================================
# Load Blender environment_config.py
# ============================================================
def load_environment():
    """Load the Blender-exported environment_config.py"""
    path = os.path.abspath("config/environment_config.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Environment config not found: {path}")

    env = {}
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()
        exec(code, {}, env)

    return env["environment_config"]

# ============================================================
# Mapping DISPLAY → ScreenSimulatorSet name
# ============================================================
DISPLAY_MAP = {
    "\\\\.\\DISPLAY2": "ScreenSimulatorSet_1",
    "\\\\.\\DISPLAY3": "ScreenSimulatorSet_2",
    "\\\\.\\DISPLAY4": "ScreenSimulatorSet_3",
}

# ============================================================
# Utility functions
# ============================================================
def normalize(v):
    v = np.array(v, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm < 1e-7:
        return v
    return v / norm

def intersect_ray_pointcloud(ray_origin, ray_dir, points, max_distance=0.05):
    """
    Ray → 点群の最近点（近似スクリーン・鏡）
    """
    diffs = points - ray_origin
    t = np.dot(diffs, ray_dir)
    valid = t > 0
    if not np.any(valid):
        return None

    diffs = diffs[valid]
    t = t[valid]
    proj = ray_origin + ray_dir * t[:, None]
    dist = np.linalg.norm(points[valid] - proj, axis=1)

    idx = np.argmin(dist)
    if dist[idx] > max_distance:
        return None

    return points[valid][idx]

# ============================================================
# Ray Tracing warp_map generator
# ============================================================
def compute_warp_map(projector, mirror_points, screen_points, src_size):
    width, height = src_size
    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)

    origin = np.array(projector["origin"], dtype=np.float32)
    direction = normalize(projector["direction"])

    fov_h = radians(projector["fov_h"])
    fov_v = radians(projector["fov_v"])

    x_lin = np.linspace(-tan(fov_h / 2), tan(fov_h / 2), width)
    y_lin = np.linspace(-tan(fov_v / 2), tan(fov_v / 2), height)

    mirror_center = mirror_points.mean(axis=0)

    for iy, fy in enumerate(y_lin):
        for ix, fx in enumerate(x_lin):
            # ① プロジェクター画素からのレイ
            ray_dir = normalize(direction + np.array([fx, fy, 0], dtype=np.float32))

            # ② 鏡点群に対して衝突
            hit_mirror = intersect_ray_pointcloud(origin, ray_dir, mirror_points)
            if hit_mirror is None:
                continue

            # 法線ベクトル（鏡中心向き）
            N = normalize(hit_mirror - mirror_center)

            # ③ 反射レイ
            ray_reflect = ray_dir - 2 * np.dot(ray_dir, N) * N

            # ④ スクリーン点群に衝突
            hit_screen = intersect_ray_pointcloud(hit_mirror, ray_reflect, screen_points)
            if hit_screen is None:
                continue

            sx, sy, _ = hit_screen
            map_x[iy, ix] = sx
            map_y[iy, ix] = sy

    return {"mode": "warp_map", "map_x": map_x, "map_y": map_y}

# ============================================================
# prepare_warp() – media_player_multi から呼ばれる
# ============================================================
def prepare_warp(display_name, mode, src_size, load_points_func=None, log_func=None):
    def log(msg):
        if log_func:
            log_func(msg)
        else:
            print(msg)

    cache_key = (display_name, mode, src_size)
    if cache_key in warp_cache:
        return warp_cache[cache_key]

    env = load_environment()

    if display_name in DISPLAY_MAP:
        set_name = DISPLAY_MAP[display_name]

        target_set = next((s for s in env["screen_simulation_sets"] if s["name"] == set_name), None)
        if target_set is None:
            log(f"[ERROR] No matching set for {display_name}")
            return None

        projector = target_set["projector"]
        mirror_pts = np.array(target_set["mirror"]["vertices"], dtype=np.float32)
        screen_pts = np.array(target_set.get("screen", {}).get("vertices", []), dtype=np.float32)

        if screen_pts.size == 0:
            log("[WARN] This set has no screen data")
            return None

        log(f"[A方式] RayTracing warp map for {display_name}")
        warp_info = compute_warp_map(projector, mirror_pts, screen_pts, src_size)
        warp_cache[cache_key] = warp_info
        return warp_info

    # fallback grid
    log(f"[Grid] fallback warp for {display_name}")
    if load_points_func:
        pts = load_points_func(display_name, mode)
    else:
        return None

    if pts is None or len(pts) < 4:
        log(f"[WARN] No grid for {display_name}")
        return None

    pts = np.array(pts, dtype=np.float32)
    w, h = src_size

    if mode == "perspective":
        src_pts = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_pts, pts[:4])
        warp_cache[cache_key] = {"mode":"perspective", "matrix":M}
        return warp_cache[cache_key]

    elif mode == "warp_map":
        ys, xs = np.indices((h, w), dtype=np.float32)
        warp_cache[cache_key] = {"mode":"warp_map", "map_x":xs, "map_y":ys}
        return warp_cache[cache_key]

    return None

# ============================================================
# Warp apply
# ============================================================
def warp_image(image, warp_info, log_func=None):
    if image is None or warp_info is None:
        return image

    h, w = image.shape[:2]

    if warp_info["mode"] == "perspective":
        return cv2.warpPerspective(image, warp_info["matrix"], (w, h))

    elif warp_info["mode"] == "warp_map":
        return cv2.remap(image, warp_info["map_x"], warp_info["map_y"],
                         interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return image
