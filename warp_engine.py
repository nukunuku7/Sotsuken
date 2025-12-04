# warp_engine.py (GPU対応版)
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

# --- GPU 検出: OpenCV CUDA と CuPy を試す --------------------------------
USE_CV2_CUDA = False
USE_CUPY = False
try:
    if hasattr(cv2, "cuda"):
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                USE_CV2_CUDA = True
        except Exception:
            USE_CV2_CUDA = False
except Exception:
    USE_CV2_CUDA = False

try:
    import cupy as cp
    # 簡易チェック: デバイスが使えるか
    try:
        _ = cp.cuda.Device().id
        USE_CUPY = True
    except Exception:
        USE_CUPY = False
except Exception:
    USE_CUPY = False

# Prepare a CuPy kernel for remap (bilinear) if cupy is available
CUPY_REMAP_KERNEL = None
if USE_CUPY:
    # CUDA C kernel: bilinear sampling from src using floating map coords (map_x,map_y).
    # src: uint8 pointer (h_src x w_src x c), stored as linear array
    # map_x,map_y: float arrays (h_dst x w_dst) with source x,y (float)
    # dst: uint8 pointer (h_dst x w_dst x c)
    # We'll handle bounds checking: if sampled coords outside -> write black.
    remap_code = r'''
    extern "C" __global__
    void remap_bilinear(const unsigned char* src, int h_src, int w_src, int c,
                        const float* map_x, const float* map_y,
                        unsigned char* dst, int h_dst, int w_dst) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= w_dst || y >= h_dst) return;
        int dst_idx = (y * w_dst + x) * c;

        int idx = y * w_dst + x;
        float fx = map_x[idx];
        float fy = map_y[idx];

        // treat out-of-range as black
        if (!(fx >= 0.0f && fx <= (float)(w_src - 1) && fy >= 0.0f && fy <= (float)(h_src - 1))) {
            for (int ch = 0; ch < c; ++ch) dst[dst_idx + ch] = 0;
            return;
        }

        // bilinear
        int x0 = (int)floorf(fx);
        int y0 = (int)floorf(fy);
        int x1 = min(x0 + 1, w_src - 1);
        int y1 = min(y0 + 1, h_src - 1);

        float wx = fx - (float)x0;
        float wy = fy - (float)y0;

        for (int ch = 0; ch < c; ++ch) {
            int idx00 = (y0 * w_src + x0) * c + ch;
            int idx10 = (y0 * w_src + x1) * c + ch;
            int idx01 = (y1 * w_src + x0) * c + ch;
            int idx11 = (y1 * w_src + x1) * c + ch;

            float v00 = (float)src[idx00];
            float v10 = (float)src[idx10];
            float v01 = (float)src[idx01];
            float v11 = (float)src[idx11];

            float v0 = v00 * (1.0f - wx) + v10 * wx;
            float v1 = v01 * (1.0f - wx) + v11 * wx;
            float v = v0 * (1.0f - wy) + v1 * wy;

            int out = (int)(v + 0.5f);
            if (out < 0) out = 0;
            if (out > 255) out = 255;
            dst[dst_idx + ch] = (unsigned char) out;
        }
    }
    '''
    try:
        CUPY_REMAP_KERNEL = cp.RawKernel(remap_code, 'remap_bilinear')
    except Exception:
        CUPY_REMAP_KERNEL = None
        USE_CUPY = False

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
    t_vals = t_vals[mask]
    candidate_pts = pts[mask]
    ts = t_vals
    projected = ray_o[None, :] + np.outer(ts, ray_d)
    dists = np.linalg.norm(projected - candidate_pts, axis=1)
    idx = np.argmin(dists)
    best_t = ts[idx]
    best_pt = candidate_pts[idx]
    best_dist = dists[idx]
    if best_t < t_min or best_t > t_max:
        return None, None, None
    return best_pt, float(best_t), float(best_dist)

def _estimate_normal_at_vertex(vertex_idx, pts, k=20):
    pts = np.asarray(pts, dtype=np.float64)
    v = pts[vertex_idx]
    dists = np.linalg.norm(pts - v[None, :], axis=1)
    idxs = np.argsort(dists)
    k = min(k, len(idxs))
    neigh = pts[idxs[:k]]
    centroid = neigh.mean(axis=0)
    cov = np.cov((neigh - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]
    return _normalize(normal)

def _estimate_normals_for_pointcloud(pts, sample_stride=1, k=20):
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    normals = np.zeros_like(pts)
    for i in range(0, n, sample_stride):
        dists = np.linalg.norm(pts - pts[i], axis=1)
        idn = np.argsort(dists)[:min(k, n)]
        neigh = pts[idn]
        centroid = neigh.mean(axis=0)
        cov = np.cov((neigh - centroid).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]
        normals[i] = _normalize(normal)
    for i in range(n):
        if np.all(normals[i] == 0):
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

# --- perspective matrix generator ---
def generate_perspective_matrix(src_size, dst_points):
    w, h = src_size
    if len(dst_points) != 4:
        raise ValueError("射影変換には4点が必要です")
    src_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    dst_pts = np.array(dst_points, dtype=np.float32)
    return cv2.getPerspectiveTransform(src_pts, dst_pts)

# DISPLAY_TO_SIMSET (変更しない)
DISPLAY_TO_SIMSET = {
    r"\\.\DISPLAY2": 0,
    r"\\.\DISPLAY3": 1,
    r"\\.\DISPLAY4": 2,
}

# --- prepare_warp: 既存コード (省略せずそのまま使用) ---
def prepare_warp(display_name, mode, src_size, load_points_func=None, log_func=None):
    _log(f"[prepare_warp] Display={display_name} Mode={mode} Size={src_size}", log_func)
    cache_key = (display_name, mode, src_size)
    if cache_key in warp_cache:
        _log("[cache] hit", log_func)
        return warp_cache[cache_key]

    if mode == "perspective":
        if load_points_func:
            pts = load_points_func(display_name, mode)
        else:
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

    # warp_map mode: heavy CPU precompute (unchanged)
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

    proj_origin = np.array(proj["origin"], dtype=np.float64)
    proj_dir = _normalize(np.array(proj["direction"], dtype=np.float64))
    fov_h = float(proj.get("fov_h", 53.13))
    fov_v = float(proj.get("fov_v", fov_h))
    proj_resolution = tuple(proj.get("resolution", [int(src_size[0]), int(src_size[1])]))

    keystone_v = float(proj.get("keystone_v", 0.0))
    if abs(keystone_v) > 1e-6:
        default_up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(default_up, proj_dir)) > 0.99:
            default_up = np.array([0.0, 1.0, 0.0])
        right = _normalize(np.cross(proj_dir, default_up))
        up = _normalize(np.cross(right, proj_dir))
        theta = math.radians(keystone_v)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        proj_dir = _normalize(
            proj_dir * cos_t +
            np.cross(right, proj_dir) * sin_t +
            right * np.dot(right, proj_dir) * (1 - cos_t)
        )
        _log(f"[keystone] vertical keystone applied: {keystone_v} deg", log_func)

    mirror_pts = np.array(mirror.get("vertices", []), dtype=np.float64)
    screen_pts = np.array(screen.get("vertices", []), dtype=np.float64)
    if mirror_pts.size == 0 or screen_pts.size == 0:
        _log("[WARN] mirror or screen point cloud empty", log_func)
        return None

    screen_centroid, screen_u, screen_v, screen_normal = _fit_plane(screen_pts)
    uv_coords = np.stack([np.dot(screen_pts - screen_centroid, screen_u),
                          np.dot(screen_pts - screen_centroid, screen_v)], axis=1)
    u_min, v_min = uv_coords.min(axis=0)
    u_max, v_max = uv_coords.max(axis=0)

    w_out, h_out = int(src_size[0]), int(src_size[1])
    map_x = np.zeros((h_out, w_out), dtype=np.float32)
    map_y = np.zeros((h_out, w_out), dtype=np.float32)

    try:
        mirror_normals = _estimate_normals_for_pointcloud(mirror_pts, sample_stride=1, k=16)
    except Exception:
        mirror_normals = np.tile(np.array([0,0,1.0]), (len(mirror_pts),1))

    fov_h_rad = math.radians(fov_h)
    fov_v_rad = math.radians(fov_v)

    default_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(default_up, proj_dir)) > 0.99:
        default_up = np.array([0.0, 1.0, 0.0])
    right = _normalize(np.cross(proj_dir, default_up))
    up = _normalize(np.cross(right, proj_dir))

    for yy in range(h_out):
        v = ( (yy + 0.5) / h_out - 0.5 ) * fov_v_rad
        for xx in range(w_out):
            u = ( (xx + 0.5) / w_out - 0.5 ) * fov_h_rad
            dir_cam = (_normalize(proj_dir) * 1.0 +
                       right * math.tan(u) +
                       up * math.tan(v))
            dir_cam = _normalize(dir_cam)

            mirror_hit_pt, t_m, dist_m = _nearest_along_ray(proj_origin, dir_cam, mirror_pts, t_min=0.01, t_max=10.0)
            if mirror_hit_pt is None:
                map_x[yy, xx] = 0.0
                map_y[yy, xx] = 0.0
                continue

            diffs = mirror_pts - mirror_hit_pt[None,:]
            idx = int(np.argmin(np.linalg.norm(diffs, axis=1)))
            n = mirror_normals[idx]
            if np.linalg.norm(n) == 0:
                n = screen_normal

            refl = _reflect(dir_cam, n)
            screen_hit_pt, t_s, dist_s = _nearest_along_ray(mirror_hit_pt + refl * 1e-6, refl, screen_pts, t_min=0.01, t_max=10.0)
            if screen_hit_pt is None:
                map_x[yy, xx] = 0.0
                map_y[yy, xx] = 0.0
                continue

            rel = screen_hit_pt - screen_centroid
            ucoord = float(np.dot(rel, screen_u))
            vcoord = float(np.dot(rel, screen_v))

            if (u_max - u_min) == 0 or (v_max - v_min) == 0:
                map_x[yy, xx] = 0.0
                map_y[yy, xx] = 0.0
                continue

            fx = (ucoord - u_min) / (u_max - u_min)
            fy = (vcoord - v_min) / (v_max - v_min)

            if not (0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0):
                map_x[yy, xx] = 0.0
                map_y[yy, xx] = 0.0
            else:
                sx = fx * (proj_resolution[0] - 1)
                sy = (1.0 - fy) * (proj_resolution[1] - 1)
                map_x[yy, xx] = float(sx)
                map_y[yy, xx] = float(sy)

    warp_cache[cache_key] = {
        "mode": "warp_map",
        "map_x": map_x,
        "map_y": map_y
    }
    _log(f"[OK] warp_map prepared for {display_name} ({w_out}x{h_out})", log_func)
    return warp_cache[cache_key]

# --- warp_image: GPU を優先する実装 ---
def warp_image(image, warp_info, log_func=None):
    """
    image: HxWxC (RGB uint8) の numpy array
    warp_info: prepare_warp の戻り値
    動作:
      - perspective: 可能なら cv2.cuda.warpPerspective を使う
      - warp_map: 可能なら CuPy カーネルで remap（bilinear）を行う
      - どちらも無ければ既存の CPU 実装にフォールバック
    """
    if image is None or warp_info is None:
        return image

    h, w = image.shape[:2]

    try:
        mode = warp_info.get("mode", None)
        if mode == "perspective":
            matrix = warp_info.get("matrix", None)
            if matrix is None:
                return image

            # === GPU path for perspective ===
            if USE_CV2_CUDA:
                try:
                    gpu_src = cv2.cuda_GpuMat()
                    gpu_src.upload(image)
                    gpu_dst = cv2.cuda_GpuMat()
                    # Note: warpPerspective size is (width, height) in OpenCV call
                    gpu_matrix = matrix
                    cv2.cuda.warpPerspective(gpu_src, gpu_dst, gpu_matrix, (w, h),
                                             flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_CONSTANT,
                                             borderValue=(0, 0, 0))
                    out = gpu_dst.download()
                    return out
                except Exception as e:
                    _log(f"[WARN] cv2.cuda.warpPerspective failed, falling back to CPU: {e}", log_func)
                    # fallthrough to CPU

            # CPU fallback
            warped = cv2.warpPerspective(image, matrix, (w, h),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(0, 0, 0))
            return warped

        elif mode == "warp_map":
            map_x = warp_info.get("map_x", None)
            map_y = warp_info.get("map_y", None)
            if map_x is None or map_y is None:
                return image

            # resize maps to current image size if necessary (CPU small op)
            if map_x.shape != (h, w):
                map_x = cv2.resize(map_x, (w, h), interpolation=cv2.INTER_LINEAR)
                map_y = cv2.resize(map_y, (w, h), interpolation=cv2.INTER_LINEAR)

            # === GPU path for warpmapping using CuPy kernel ===
            if USE_CUPY and CUPY_REMAP_KERNEL is not None:
                try:
                    # upload src as uint8 1D linear array
                    src_cp = cp.asarray(image)  # shape (h,w,c), dtype=uint8
                    # ensure contiguous
                    src_cp = src_cp.astype(cp.uint8, copy=False)
                    mapx_cp = cp.asarray(map_x.astype(np.float32))
                    mapy_cp = cp.asarray(map_y.astype(np.float32))

                    h_dst, w_dst = mapx_cp.shape
                    c = int(src_cp.shape[2]) if src_cp.ndim == 3 else 1

                    # flatten src and dst buffers
                    src_flat = src_cp.ravel()
                    dst_cp = cp.zeros((h_dst, w_dst, c), dtype=cp.uint8)
                    dst_flat = dst_cp.ravel()

                    threads_x = 16
                    threads_y = 16
                    block = (threads_x, threads_y, 1)
                    grid_x = (w_dst + threads_x - 1) // threads_x
                    grid_y = (h_dst + threads_y - 1) // threads_y
                    grid = (grid_x, grid_y, 1)

                    # Launch kernel
                    CUPY_REMAP_KERNEL(grid, block,
                                      (src_flat, np.int32(src_cp.shape[0]), np.int32(src_cp.shape[1]), np.int32(c),
                                       mapx_cp, mapy_cp, dst_flat, np.int32(h_dst), np.int32(w_dst)))

                    out = cp.asnumpy(dst_cp)
                    return out
                except Exception as e:
                    _log(f"[WARN] CuPy remap kernel failed, falling back to CPU remap: {e}", log_func)
                    # fallthrough to CPU

            # CPU fallback: use cv2.remap
            warped = cv2.remap(image,
                               map_x.astype(np.float32),
                               map_y.astype(np.float32),
                               interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0))
            return warped

        else:
            return image

    except Exception as e:
        if log_func:
            log_func(f"[ERROR] warp_image failed: {e}")
        else:
            print(f"[ERROR] warp_image failed: {e}")
        return image
