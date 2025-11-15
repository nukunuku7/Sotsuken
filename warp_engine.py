# warp_engine.py
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

# --- ヘルパー ---
def _log(msg, log_func=None):
    if log_func:
        log_func(msg)
    else:
        print(msg)

def _normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def _fit_plane(points):
    """
    点群から平面の基底を推定する（PCA）
    returns (origin, u_vec, v_vec, normal)
    """
    pts = np.asarray(points, dtype=np.float64)
    centroid = pts.mean(axis=0)
    cov = np.cov((pts - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigvals ascending -> smallest is normal
    normal = eigvecs[:, 0]
    u = eigvecs[:, 2]
    v = eigvecs[:, 1]
    return centroid, _normalize(u), _normalize(v), _normalize(normal)

def _nearest_along_ray(ray_o, ray_d, points, t_min=0.0, t_max=10.0):
    """
    点群 points に対し、ray = ray_o + t * ray_d が最も近づく点を探す（t>0）。
    返り値: (best_point, best_t, distance)
    """
    pts = np.asarray(points, dtype=np.float64)
    # vectors from origin to pts
    vecs = pts - ray_o[None, :]
    # project onto ray_d
    t_vals = np.dot(vecs, ray_d)
    # clamp to positive
    mask = t_vals > t_min
    if not np.any(mask):
        return None, None, None
    t_vals = t_vals[mask]
    candidate_pts = pts[mask]
    ts = t_vals
    projected = ray_o[None, :] + np.outer(ts, ray_d)
    dists = np.linalg.norm(projected - candidate_pts, axis=1)
    # find minimal distance
    idx = np.argmin(dists)
    best_t = ts[idx]
    best_pt = candidate_pts[idx]
    best_dist = dists[idx]
    if best_t < t_min or best_t > t_max:
        return None, None, None
    return best_pt, float(best_t), float(best_dist)

def _estimate_normal_at_vertex(vertex_idx, pts, k=20):
    """
    単一頂点周りの局所法線を PCA で推定する（k 近傍を採る）。
    pts は Nx3 配列。vertex_idx は pts 中の index。
    """
    pts = np.asarray(pts, dtype=np.float64)
    v = pts[vertex_idx]
    # compute distances
    dists = np.linalg.norm(pts - v[None, :], axis=1)
    idxs = np.argsort(dists)
    k = min(k, len(idxs))
    neigh = pts[idxs[:k]]
    centroid = neigh.mean(axis=0)
    cov = np.cov((neigh - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]  # smallest eigenvalue direction
    return _normalize(normal)

def _estimate_normals_for_pointcloud(pts, sample_stride=1, k=20):
    """
    点群全体の局所法線を返す（重い）。必要ならサブサンプリングを検討。
    戻り値: Nx3 normals
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    normals = np.zeros_like(pts)
    for i in range(0, n, sample_stride):
        # find nearest neighbors
        dists = np.linalg.norm(pts - pts[i], axis=1)
        idn = np.argsort(dists)[:min(k, n)]
        neigh = pts[idn]
        centroid = neigh.mean(axis=0)
        cov = np.cov((neigh - centroid).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]
        normals[i] = _normalize(normal)
    # fill missing by simple copy (for indices skipped by stride)
    for i in range(n):
        if np.all(normals[i] == 0):
            # nearest nonzero
            nn = np.where(np.linalg.norm(normals, axis=1) > 0)[0]
            if nn.size:
                normals[i] = normals[nn[0]]
            else:
                normals[i] = np.array([0,0,1])
    return normals

def _reflect(d, n):
    d = _normalize(d)
    n = _normalize(n)
    return d - 2.0 * np.dot(d, n) * n

# --- perspective 行列生成 (従来機能) ---
def generate_perspective_matrix(src_size, dst_points):
    w, h = src_size
    if len(dst_points) != 4:
        raise ValueError("射影変換には4点が必要です")
    src_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    dst_pts = np.array(dst_points, dtype=np.float32)
    return cv2.getPerspectiveTransform(src_pts, dst_pts)

# --- DISPLAY -> simulation set mapping (environment_config 側の順序にあわせて調整してください) ---
# ここは main.py と一致するように必要に応じて変更してください
DISPLAY_TO_SIMSET = {
    r"\\.\DISPLAY2": 0,  # 右
    r"\\.\DISPLAY3": 1,  # 中央
    r"\\.\DISPLAY4": 2,  # 左
}

# --- prepare_warp: main entrypoint ---
def prepare_warp(display_name, mode, src_size, load_points_func=None, log_func=None):
    """
    display_name: '\\\\.\\DISPLAY2' のような PyQt 上の名前
    mode: 'perspective' or 'warp_map'
    src_size: (width, height) - 出力先（プロジェクター側）ピクセルサイズ（例: (1920,1080)）
    load_points_func: 外部からグリッド点を供給する関数( name, mode ) -> points list
    log_func: ログ関数
    """
    _log(f"[prepare_warp] Display={display_name} Mode={mode} Size={src_size}", log_func)

    cache_key = (display_name, mode, src_size)
    if cache_key in warp_cache:
        _log("[cache] hit", log_func)
        return warp_cache[cache_key]

    # --- 1) perspective モード（既存グリッド4点） の場合は従来通り処理 ---
    if mode == "perspective":
        if load_points_func:
            pts = load_points_func(display_name, mode)
        else:
            # default: read from config/projector_profiles file
            cfg_path = os.path.join("config", "projector_profiles", f"__._{display_name}_{mode}_points.json")
            if not os.path.exists(cfg_path):
                _log(f"[WARN] perspective grid file not found: {cfg_path}", log_func)
                return None
            with open(cfg_path, "r", encoding="utf-8") as f:
                pts = json.load(f)
        if pts is None or len(pts) < 4:
            _log("[WARN] perspective points missing", log_func)
            return None
        matrix = generate_perspective_matrix(src_size, pts[:4])
        warp_cache[cache_key] = {"mode": "perspective", "matrix": matrix}
        _log(f"[OK] perspective matrix prepared for {display_name}", log_func)
        return warp_cache[cache_key]

    # --- 2) warp_map モード（点群ベースの近似レイトレーシング） ---
    # We need environment_config available
    if environment_config is None:
        _log("[ERROR] environment_config not available. Place config/environment_config.py", log_func)
        return None

    if display_name not in DISPLAY_TO_SIMSET:
        _log(f"[WARN] {display_name} not in DISPLAY_TO_SIMSET mapping", log_func)
        return None

    sim_idx = DISPLAY_TO_SIMSET[display_name]
    try:
        sim_set = environment_config["screen_simulation_sets"][sim_idx]
    except Exception as e:
        _log(f"[ERROR] cannot access simulation set {sim_idx}: {e}", log_func)
        return None

    proj = sim_set.get("projector")
    mirror = sim_set.get("mirror")
    screen = sim_set.get("screen", None)
    if not proj or not mirror or not screen:
        _log("[WARN] incomplete sim set (need projector/mirror/screen)", log_func)
        return None

    # projector origin, direction, FOV, resolution
    proj_origin = np.array(proj["origin"], dtype=np.float64)
    proj_dir = _normalize(np.array(proj["direction"], dtype=np.float64))
    fov_h = float(proj.get("fov_h", 53.13))
    fov_v = float(proj.get("fov_v", fov_h))
    proj_resolution = tuple(proj.get("resolution", [int(src_size[0]), int(src_size[1])]))

    # mirror point cloud & screen point cloud
    mirror_pts = np.array(mirror.get("vertices", []), dtype=np.float64)
    screen_pts = np.array(screen.get("vertices", []), dtype=np.float64)
    if mirror_pts.size == 0 or screen_pts.size == 0:
        _log("[WARN] mirror or screen point cloud empty", log_func)
        return None

    # precompute screen plane basis for mapping 3D hit -> 2D pixel
    screen_centroid, screen_u, screen_v, screen_normal = _fit_plane(screen_pts)
    # project screen_pts into (u,v) coordinates for bounding box
    uv_coords = np.stack([np.dot(screen_pts - screen_centroid, screen_u),
                          np.dot(screen_pts - screen_centroid, screen_v)], axis=1)
    u_min, v_min = uv_coords.min(axis=0)
    u_max, v_max = uv_coords.max(axis=0)

    w_out, h_out = int(src_size[0]), int(src_size[1])
    map_x = np.zeros((h_out, w_out), dtype=np.float32)
    map_y = np.zeros((h_out, w_out), dtype=np.float32)

    # For speed: optionally subsample mirror normals or compute normals for points
    # We'll compute normals for mirror pointcloud with modest k to get local normal estimates.
    try:
        mirror_normals = _estimate_normals_for_pointcloud(mirror_pts, sample_stride=1, k=16)
    except Exception:
        mirror_normals = np.tile(np.array([0,0,1.0]), (len(mirror_pts),1))

    # define projector imaging plane: we will map pixel (i,j) in output (src_size) -> ray
    # Approach: assume projector is pinhole at proj_origin, with horizontal FOV fov_h covering width w_out, vertical fov_v covering height h_out.
    # pixel center angle offsets:
    # map pixel x in [0..w_out-1] -> angle_x = ( (x+0.5)/w_out - 0.5) * fov_h (deg -> rad)
    # same for y.
    fov_h_rad = math.radians(fov_h)
    fov_v_rad = math.radians(fov_v)

    # Precompute image plane basis for projector. We need two orthonormal axes perpendicular to proj_dir.
    # pick arbitrary up (0,0,1) unless parallel
    default_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(default_up, proj_dir)) > 0.99:
        default_up = np.array([0.0, 1.0, 0.0])
    right = _normalize(np.cross(proj_dir, default_up))
    up = _normalize(np.cross(right, proj_dir))

    # For each pixel compute ray direction
    # We'll iterate over pixels - this is the heavy part. Consider subsampling for speed if too slow.
    for yy in range(h_out):
        # vertical angle
        v = ( (yy + 0.5) / h_out - 0.5 ) * fov_v_rad
        for xx in range(w_out):
            u = ( (xx + 0.5) / w_out - 0.5 ) * fov_h_rad
            # direction in camera coordinates
            dir_cam = (_normalize(proj_dir) * 1.0 +
                       right * math.tan(u) +
                       up * math.tan(v))
            dir_cam = _normalize(dir_cam)

            # 1) intersect with mirror (approx by nearest point along ray)
            mirror_hit_pt, t_m, dist_m = _nearest_along_ray(proj_origin, dir_cam, mirror_pts, t_min=0.01, t_max=10.0)
            if mirror_hit_pt is None:
                # leave mapping at 0 (black)
                map_x[yy, xx] = 0.0
                map_y[yy, xx] = 0.0
                continue

            # find index of that mirror point to get normal
            # (brute-force find index; could optimize with kd-tree)
            # use exact match by distance
            diffs = mirror_pts - mirror_hit_pt[None,:]
            idx = int(np.argmin(np.linalg.norm(diffs, axis=1)))
            n = mirror_normals[idx]
            if np.linalg.norm(n) == 0:
                n = screen_normal  # fallback

            # reflect
            refl = _reflect(dir_cam, n)

            # 2) trace reflected ray to screen (nearest point along reflected ray)
            screen_hit_pt, t_s, dist_s = _nearest_along_ray(mirror_hit_pt + refl * 1e-6, refl, screen_pts, t_min=0.01, t_max=10.0)
            if screen_hit_pt is None:
                map_x[yy, xx] = 0.0
                map_y[yy, xx] = 0.0
                continue

            # 3) convert screen_hit_pt -> local (u,v) coordinates on screen plane
            rel = screen_hit_pt - screen_centroid
            ucoord = float(np.dot(rel, screen_u))
            vcoord = float(np.dot(rel, screen_v))

            # normalized fraction across screen bounds
            if (u_max - u_min) == 0 or (v_max - v_min) == 0:
                map_x[yy, xx] = 0.0
                map_y[yy, xx] = 0.0
                continue

            fx = (ucoord - u_min) / (u_max - u_min)
            fy = (vcoord - v_min) / (v_max - v_min)

            # clamp
            if not (0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0):
                # outside screen bounds -> mark transparent/black
                map_x[yy, xx] = 0.0
                map_y[yy, xx] = 0.0
            else:
                # map to pixel coords in the projector's image plane (assume projector resolution)
                sx = fx * (proj_resolution[0] - 1)
                sy = (1.0 - fy) * (proj_resolution[1] - 1)  # v coordinate -> image y (flip if needed)
                # However prepare_warp should return mapping from dest pixel to source pixel coordinates.
                # Since our dest is the projector (src_size), map_x,map_y use source image coordinate system (projector image)
                map_x[yy, xx] = float(sx)
                map_y[yy, xx] = float(sy)

    warp_cache[cache_key] = {
        "mode": "warp_map",
        "map_x": map_x,
        "map_y": map_y
    }
    _log(f"[OK] warp_map prepared for {display_name} ({w_out}x{h_out})", log_func)
    return warp_cache[cache_key]

# --- warp_image: same as before (apply mapping) ---
def warp_image(image, warp_info, log_func=None):
    if image is None or warp_info is None:
        return image

    h, w = image.shape[:2]
    try:
        if warp_info["mode"] == "perspective":
            warped = cv2.warpPerspective(image, warp_info["matrix"], (w, h),
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        elif warp_info["mode"] == "warp_map":
            # map_x,map_y are float coordinates in source image (projector image)
            map_x = warp_info["map_x"]
            map_y = warp_info["map_y"]
            # resize maps to current image size if necessary
            if map_x.shape != (h, w):
                map_x = cv2.resize(map_x, (w, h), interpolation=cv2.INTER_LINEAR)
                map_y = cv2.resize(map_y, (w, h), interpolation=cv2.INTER_LINEAR)
            warped = cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32),
                               interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        else:
            return image

        return warped

    except Exception as e:
        if log_func:
            log_func(f"[ERROR] warp_image failed: {e}")
        else:
            print(f"[ERROR] warp_image failed: {e}")
        return image
