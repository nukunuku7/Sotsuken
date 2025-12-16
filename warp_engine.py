# warp_engine.py (GPU対応版 + 転送最適化)
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

def _log(msg, log_func=None):
    """ログを出力するための共通関数"""
    if log_func:
        try:
            log_func(msg)
        except Exception:
            print(msg)
    else:
        print(msg)

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
    try:
        dev = cp.cuda.Device()
        _log(f"[CuPy] GPU device detected: {dev.id}")
        USE_CUPY = True
    except Exception as e:
        _log(f"[CuPy] GPU check failed: {e}")
        USE_CUPY = False
except Exception as e:
    _log(f"[CuPy] import failed: {e}")
    USE_CUPY = False

# Prepare a CuPy kernel for remap (bilinear) if cupy is available
CUPY_REMAP_KERNEL = None
if USE_CUPY:
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

        if (!(fx >= 0.0f && fx <= (float)(w_src - 1) && fy >= 0.0f && fy <= (float)(h_src - 1))) {
            for (int ch = 0; ch < c; ++ch) dst[dst_idx + ch] = 0;
            return;
        }

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
    except Exception as e:
        _log(f"[CuPy] RawKernel compile failed: {e}")
        CUPY_REMAP_KERNEL = None
        USE_CUPY = False


# --- GPU キャッシュ & 設定 -----------------------------------------------
_gpu_cache = {
    "cv2_src": None,       # cv2.cuda_GpuMat reuse (perspective)
    "cv2_dst": None,
    "cupy_map_x": None,    # map_x on GPU
    "cupy_map_y": None,    # map_y on GPU
    "cupy_src": None,      # GPU src buffer (device)
    "cupy_dst": None,      # GPU dst buffer (device)
    "cupy_pinned": None,   # pinned host memory (memoryview)
    "last_shape": None,    # shape used for pinned buffer
    "stream": None,        # primary cupy stream for async ops
}

# default stream init when cupy available
if USE_CUPY:
    try:
        _gpu_cache["stream"] = cp.cuda.Stream(non_blocking=True)
    except Exception:
        _gpu_cache["stream"] = None

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
DISPLAY_TO_SIMSET = {}

try:
    with open("config/display_map.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        DISPLAY_TO_SIMSET = data.get("display_map", {})
except Exception:
    DISPLAY_TO_SIMSET = {}


# --- prepare_warp: 既存コード (省略せずそのまま使用) ---
def prepare_warp(display_name, mode, src_size, load_points_func=None, log_func=None):
    # sanitize display name for file paths
    safe_name = (
        display_name.replace("(", "_")
                    .replace(")", "_")
                    .replace(" ", "_")
                    .replace("-", "_")
                    .replace("/", "_")
    )

    display_name = safe_name
    
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

        # ★★★ 修正：matrix から map_x と map_y を生成する ★★★
        w_out, h_out = int(src_size[0]), int(src_size[1])
        
        # 恒等写像 (Identity Map) を作成
        map_x, map_y = np.indices((h_out, w_out), dtype=np.float32)
        map_x = map_x.T
        map_y = map_y.T

        # matrix を使って map_x と map_y を変換（OpenCVの機能で remap のマップを作る）
        map_x, map_y = cv2.convertMaps(
            cv2.warpPerspective(map_x, matrix, src_size, flags=cv2.INTER_LINEAR),
            cv2.warpPerspective(map_y, matrix, src_size, flags=cv2.INTER_LINEAR),
            cv2.CV_32FC1
        )
        
        # warp_cache[cache_key] = {"mode": "perspective", "matrix": matrix} # 辞書のキャッシュは不要
        warp_cache[cache_key] = (map_x, map_y) # キャッシュを (map_x, map_y) のタプルにする
        
        _log(f"[OK] perspective matrix prepared for {display_name}", log_func)
        return map_x, map_y # ★ map_x, map_y のタプルを直接返す ★

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
    return map_x, map_y

# --- warp_image: GPU を優先する実装 + 転送最適化 --------------
def warp_image(image, warp_info, log_func=None):
    """
    image: HxWxC (RGB uint8) の numpy array
    warp_info: prepare_warp の戻り値
    """
    if image is None or warp_info is None:
        return image

    h, w = image.shape[:2]

    try:
        mode = warp_info.get("mode", None)
        # -------------------
        # perspective
        # -------------------
        if mode == "perspective":
            matrix = warp_info.get("matrix", None)
            if matrix is None:
                return image

            # use cv2.cuda if available, reuse GpuMat
            if USE_CV2_CUDA:
                try:
                    if _gpu_cache["cv2_src"] is None:
                        _gpu_cache["cv2_src"] = cv2.cuda_GpuMat()
                        _gpu_cache["cv2_dst"] = cv2.cuda_GpuMat()
                    gsrc = _gpu_cache["cv2_src"]
                    gdst = _gpu_cache["cv2_dst"]

                    # upload (cv2 handles internal pinned/fast path)
                    gsrc.upload(image)
                    cv2.cuda.warpPerspective(gsrc, gdst, matrix, (w, h),
                                             flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_CONSTANT,
                                             borderValue=(0, 0, 0))
                    out = gdst.download()
                    return out
                except Exception as e:
                    _log(f"[WARN] cv2.cuda perspective failed, fallback to CPU: {e}", log_func)
                    # fallthrough

            # CPU fallback
            warped = cv2.warpPerspective(image, matrix, (w, h),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(0, 0, 0))
            return warped

        # -------------------
        # warp_map
        # -------------------
        elif mode == "warp_map":
            map_x = warp_info.get("map_x", None)
            map_y = warp_info.get("map_y", None)
            if map_x is None or map_y is None:
                return image

            # resize maps to current image size if necessary (cheap)
            if map_x.shape != (h, w):
                map_x = cv2.resize(map_x, (w, h), interpolation=cv2.INTER_LINEAR)
                map_y = cv2.resize(map_y, (w, h), interpolation=cv2.INTER_LINEAR)

            # GPU path using CuPy kernel + pinned memory + async stream
            if USE_CUPY and CUPY_REMAP_KERNEL is not None:
                try:
                    stream = _gpu_cache.get("stream", None)
                    if stream is None:
                        stream = cp.cuda.Stream(non_blocking=True)

                    # cache map_x/map_y on GPU (once)
                    if _gpu_cache["cupy_map_x"] is None or _gpu_cache["cupy_map_x"].shape != map_x.shape:
                        _gpu_cache["cupy_map_x"] = cp.asarray(map_x.astype(np.float32))
                        _gpu_cache["cupy_map_y"] = cp.asarray(map_y.astype(np.float32))

                    mapx_cp = _gpu_cache["cupy_map_x"]
                    mapy_cp = _gpu_cache["cupy_map_y"]

                    # prepare pinned buffer if needed
                    nbytes = image.nbytes
                    if _gpu_cache["cupy_pinned"] is None or _gpu_cache["last_shape"] != image.shape:
                        # allocate pinned memory and create a memoryview
                        mem = cp.cuda.alloc_pinned_memory(nbytes)
                        _gpu_cache["cupy_pinned"] = mem
                        _gpu_cache["last_shape"] = image.shape

                        # allocate device src/dst buffers sized to image
                        _gpu_cache["cupy_src"] = cp.empty(image.shape, dtype=cp.uint8)
                        _gpu_cache["cupy_dst"] = cp.empty_like(_gpu_cache["cupy_src"])

                    # copy into pinned memory (host mem)
                    # mem supports buffer protocol in recent cupy; use memoryview
                    buf = memoryview(_gpu_cache["cupy_pinned"])
                    buf[:] = image.tobytes()

                    # async copy pinned -> device
                    src_dev = _gpu_cache["cupy_src"]
                    dst_dev = _gpu_cache["cupy_dst"]
                    # use stream for async copy and kernel
                    with stream:
                        # asynchronous copy from pinned host to device
                        cp.cuda.runtime.memcpyAsync(
                            src_dev.data.ptr,
                            cp.cuda.runtime.get_device_pointer(buf),
                            nbytes,
                            cp.cuda.runtime.cudaMemcpyHostToDevice,
                            stream.ptr
                        )

                        # launch kernel (grid/block choice)
                        h_dst, w_dst = image.shape[0], image.shape[1]
                        c = image.shape[2] if image.ndim == 3 else 1
                        threads_x = 16
                        threads_y = 16
                        block = (threads_x, threads_y, 1)
                        grid_x = (w_dst + threads_x - 1) // threads_x
                        grid_y = (h_dst + threads_y - 1) // threads_y
                        grid = (grid_x, grid_y, 1)

                        CUPY_REMAP_KERNEL(grid, block,
                                          (src_dev.ravel(), np.int32(src_dev.shape[0]), np.int32(src_dev.shape[1]), np.int32(c),
                                           mapx_cp, mapy_cp, dst_dev.ravel(), np.int32(h_dst), np.int32(w_dst)))

                    # synchronize stream before returning data to host
                    stream.synchronize()
                    out = cp.asnumpy(dst_dev)
                    return out

                except Exception as e:
                    _log(f"[WARN] CuPy remap kernel failed, fall back to CPU remap: {e}", log_func)
                    # fallthrough to CPU

            # CPU fallback
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

# warp_engine.py の末尾に追加

def convert_maps_to_uv_texture_data(map_x, map_y, width, height):
    """
    OpenCVのmap_x, map_y (pixel単位) を
    OpenGL/ModernGL用のUVマップ (0.0~1.0正規化, float32, HxWx2) に変換する
    """
    # 正規化 (0.0 ~ 1.0)
    # OpenCVの座標系に合わせて、範囲外の処理などをここで行うことも可能
    u = map_x.astype(np.float32) / width
    v = map_y.astype(np.float32) / height
    
    # (Height, Width, 2) の形状にスタックする
    # channel 0 = u, channel 1 = v
    uv_map = np.dstack((u, v))
    
    # OpenGLのテクスチャ座標系(左下原点)と画像の座標系(左上原点)の違いを吸収するため
    # 必要に応じてYを反転するが、今回は画像自体をそのまま扱うため、
    # シェーダー側でY反転するか、ここで調整する。
    # 通常MSS取得画像とOpenCVマップの整合性を保つにはそのままで良い場合が多いが、
    # 上下逆になる場合はここを v = 1.0 - v とする。
    
    return uv_map.astype('f4').tobytes()