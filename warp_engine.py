# warp_engine.py
# 歪み補正マップの生成と、ModernGL向けUVテクスチャへの変換を行うユーティリティ

import os
import json
import math
import numpy as np
import cv2 # マップ生成（perspectiveモード）およびヘルパー関数で使用
from PyQt5.QtGui import QGuiApplication

# 環境設定ファイル (主にwarp_mapモードで使用) のインポートを試みる
try:
    from config.environment_config import environment_config
except Exception:
    environment_config = None # ファイルがない場合は None とする

# グローバルなキャッシュ辞書
warp_cache = {}

def _log(msg, log_func=None):
    """ログを出力するための共通関数。外部のロガーが利用可能ならそちらを使用。"""
    if log_func:
        try:
            log_func(msg)
        except Exception:
            print(msg)
    else:
        print(msg)

# ディスプレイとシミュレーションセットの自動割り当て関数
def auto_assign_simsets(log_func=None):
    """
    ディスプレイの左→右順に ScreenSimulatorSet_x を自動割り当て。
    DISPLAY_TO_SIMSET を更新する。
    """
    global DISPLAY_TO_SIMSET

    if environment_config is None:
        _log("[WARN] environment_config not available.", log_func)
        return

    screens = []
    try:
        screens = QGuiApplication.screens()
    except Exception:
        pass

    if not screens:
        _log("[WARN] No screens detected.", log_func)
        return

    # 左→右順
    ordered_screens = sorted(screens, key=lambda s: s.geometry().x())
    display_names = [s.name() for s in ordered_screens]

    simsets = environment_config.get("screen_simulation_sets", [])
    simset_count = len(simsets)
    if simset_count == 0:
        _log("[WARN] No screen_simulation_sets found.", log_func)
        return

    # 割り当て
    DISPLAY_TO_SIMSET = {}
    for idx, name in enumerate(display_names):
        # ディスプレイの左から順にセットを割り当て
        sim_idx = idx % simset_count
        DISPLAY_TO_SIMSET[name] = sim_idx
        _log(f"[ASSIGN] {name} -> ScreenSimulatorSet_{sim_idx+1}", log_func)

    _log(f"[INFO] Assigned {len(DISPLAY_TO_SIMSET)} screens to {simset_count} sets.", log_func)

# 起動時に自動割り当て
auto_assign_simsets()

# --- ジオメトリ計算ヘルパー関数 (warp_mapモード用) --------------------------------

def _normalize(v):
    """ベクトルを正規化する"""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def _fit_plane(points):
    """点群に最もよくフィットする平面の中心、UV軸、法線を計算する"""
    pts = np.asarray(points, dtype=np.float64)
    centroid = pts.mean(axis=0)
    # 共分散行列を計算
    cov = np.cov((pts - centroid).T)
    # 固有値分解 (最も小さい固有値に対応する固有ベクトルが法線)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]
    u = eigvecs[:, 2] # 平面内の第一軸
    v = eigvecs[:, 1] # 平面内の第二軸
    return centroid, _normalize(u), _normalize(v), _normalize(normal)

def _nearest_along_ray(ray_o, ray_d, points, t_min=0.0, t_max=10.0):
    """レイ(ray_o, ray_d)に沿った最近接点を見つける (LSS: Least Squares Sphere近似)"""
    pts = np.asarray(points, dtype=np.float64)
    vecs = pts - ray_o[None, :]
    t_vals = np.dot(vecs, ray_d)
    
    mask = t_vals > t_min
    if not np.any(mask):
        return None, None, None
        
    t_vals = t_vals[mask]
    candidate_pts = pts[mask]
    ts = t_vals
    
    # レイ上に点を投影
    projected = ray_o[None, :] + np.outer(ts, ray_d)
    # 投影点と元の点群の距離
    dists = np.linalg.norm(projected - candidate_pts, axis=1)
    
    idx = np.argmin(dists)
    best_t = ts[idx]
    best_pt = candidate_pts[idx]
    best_dist = dists[idx]
    
    if best_t < t_min or best_t > t_max:
        return None, None, None
        
    return best_pt, float(best_t), float(best_dist)

def _estimate_normals_for_pointcloud(pts, sample_stride=1, k=20):
    """点群の各点における法線を推定する (k-近傍点に基づくPCA/平面フィット)"""
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    normals = np.zeros_like(pts)
    
    for i in range(0, n, sample_stride):
        # i番目の点の k-近傍を見つける
        dists = np.linalg.norm(pts - pts[i], axis=1)
        idn = np.argsort(dists)[:min(k, n)]
        neigh = pts[idn]
        
        # 近傍点にフィットする平面の法線を計算
        centroid = neigh.mean(axis=0)
        cov = np.cov((neigh - centroid).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]
        normals[i] = _normalize(normal)
        
    # サンプルされなかった点の法線を近傍から補間（簡略化されたフォールバック）
    for i in range(n):
        if np.all(normals[i] == 0):
            nn = np.where(np.linalg.norm(normals, axis=1) > 0)[0]
            if nn.size:
                normals[i] = normals[nn[0]]
            else:
                normals[i] = np.array([0,0,1]) # 最終フォールバック
                
    return normals

def _reflect(d, n):
    """ベクトル d を 法線 n に対して反射させる"""
    d = _normalize(d)
    n = _normalize(n)
    return d - 2.0 * np.dot(d, n) * n

# --- Perspective Matrix (射影変換行列) 生成 ---
def generate_perspective_matrix(src_size, dst_points):
    """
    入力画像サイズ (src_size) と出力先の4点 (dst_points) から
    射影変換行列 (3x3 Homography Matrix) を生成する
    """
    w, h = src_size
    if len(dst_points) != 4:
        raise ValueError("射影変換には4点が必要です")
        
    # 入力画像上の四隅の座標
    src_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    # 出力画像上のターゲットの座標
    dst_pts = np.array(dst_points, dtype=np.float32)
    
    return cv2.getPerspectiveTransform(src_pts, dst_pts)


# --- 外部設定ファイルのロード ---
DISPLAY_TO_SIMSET = {}
try:
    # 外部のディスプレイとシミュレーション設定の対応マップをロード
    with open("config/display_map.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        DISPLAY_TO_SIMSET = data.get("display_map", {})
except Exception:
    DISPLAY_TO_SIMSET = {}


# --- prepare_warp: 歪み補正マップ生成のコア機能 ---
def prepare_warp(display_name, mode, src_size, src_offset_x=0, load_points_func=None, log_func=None):
    """
    指定されたディスプレイとモードに基づき、歪み補正用のマップ (map_x, map_y) を生成またはロードする。

    ※ 注意:
    - この関数は「1画面（1プロジェクター）= 1ローカル座標系」を前提とする
    - n分割（slice）やオフセット処理は GL / 描画側の責務
    - src_offset_x は設計変更により使用しない（互換性のため引数のみ残す）
    
    戻り値: (map_x, map_y) のタプル (numpy.ndarray, float32)
    """

    # ------------------------------------------------------------
    # 1. キャッシュチェック
    # ------------------------------------------------------------
    cache_key = (display_name, mode, src_size)
    if cache_key in warp_cache:
        _log("[cache] hit", log_func)
        return warp_cache[cache_key]

    w_out, h_out = int(src_size[0]), int(src_size[1])

    # ============================================================
    # Mode 1: perspective（簡易射影変換）
    # ============================================================
    if mode == "perspective":

        # --------------------------------------------------------
        # 1-1. 設定点（4点）のロード
        # --------------------------------------------------------
        if load_points_func:
            pts = load_points_func(display_name, mode)
        else:
            cfg_path = os.path.join(
                "config", "projector_profiles",
                f"__.__{display_name}_{mode}_points.json"
            )
            if not os.path.exists(cfg_path):
                _log(f"[WARN] perspective grid file not found: {cfg_path}", log_func)
                return None
            with open(cfg_path, "r", encoding="utf-8") as f:
                pts = json.load(f)

        if pts is None or len(pts) < 4:
            _log("[WARN] perspective points missing or insufficient (<4)", log_func)
            return None

        # --------------------------------------------------------
        # 1-2. 射影変換行列の生成
        # --------------------------------------------------------
        matrix = generate_perspective_matrix(src_size, pts[:4])

        # --------------------------------------------------------
        # 1-3. OpenCV remap 用マップ生成
        # --------------------------------------------------------
        # 恒等写像（出力側ピクセル座標）
        map_y_identity, map_x_identity = np.indices(
            (h_out, w_out), dtype=np.float32
        )

        # 各出力ピクセルが「元画像のどこを見るか」を計算
        map_x = cv2.warpPerspective(
            map_x_identity, matrix, src_size, flags=cv2.INTER_LINEAR
        )
        map_y = cv2.warpPerspective(
            map_y_identity, matrix, src_size, flags=cv2.INTER_LINEAR
        )

        # --------------------------------------------------------
        # ★ 旧設計（n分割前提）の名残：使用しない
        # --------------------------------------------------------
        # if src_offset_x != 0:
        #     map_x = map_x - float(src_offset_x)

        # 範囲外を安全に潰す
        map_x[map_x < 0] = 0.0
        map_x[map_x > (w_out - 1)] = 0.0
        map_y[map_y < 0] = 0.0
        map_y[map_y > (h_out - 1)] = 0.0

        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)

        # if src_offset_x != 0:
        #     map_x = map_x + float(src_offset_x)

        warp_cache[cache_key] = (map_x, map_y)
        _log(f"[OK] perspective map prepared for {display_name}", log_func)
        return map_x, map_y

    # ============================================================
    # Mode 2: warp_map（3Dシミュレーション）
    # ============================================================
    elif mode == "warp_map":

        if environment_config is None:
            _log("[ERROR] environment_config not available.", log_func)
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
        screen = sim_set.get("screen")
        if not proj or not mirror or not screen:
            _log("[WARN] incomplete sim set (need projector/mirror/screen)", log_func)
            return None

        proj_origin = np.array(proj["origin"], dtype=np.float64)
        proj_dir = _normalize(np.array(proj["direction"], dtype=np.float64))
        fov_h = float(proj.get("fov_h", 53.13))
        fov_v = float(proj.get("fov_v", fov_h))

        # ★ 1画面 = 1解像度（sliceはGL側で処理）
        proj_resolution = (w_out, h_out)

        # if src_offset_x != 0:
        #     map_x = map_x + float(src_offset_x)

        # --------------------------------------------------------
        # 以下、元ロジックそのまま（数式・構造は変更なし）
        # --------------------------------------------------------
        keystone_v = float(proj.get("keystone_v", 0.0))
        if abs(keystone_v) > 1e-6:
            default_up = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(default_up, proj_dir)) > 0.99:
                default_up = np.array([0.0, 1.0, 0.0])
            right = _normalize(np.cross(proj_dir, default_up))
            up = _normalize(np.cross(right, proj_dir))
            theta = math.radians(keystone_v)
            proj_dir = _normalize(
                proj_dir * math.cos(theta) +
                np.cross(right, proj_dir) * math.sin(theta) +
                right * np.dot(right, proj_dir) * (1 - math.cos(theta))
            )
            _log(f"[keystone] vertical keystone applied: {keystone_v} deg", log_func)

        mirror_pts = np.array(mirror.get("vertices", []), dtype=np.float64)
        screen_pts = np.array(screen.get("vertices", []), dtype=np.float64)
        if mirror_pts.size == 0 or screen_pts.size == 0:
            _log("[WARN] mirror or screen point cloud empty", log_func)
            return None

        screen_centroid, screen_u, screen_v, screen_normal = _fit_plane(screen_pts)
        uv_coords = np.stack([
            np.dot(screen_pts - screen_centroid, screen_u),
            np.dot(screen_pts - screen_centroid, screen_v)
        ], axis=1)
        u_min, v_min = uv_coords.min(axis=0)
        u_max, v_max = uv_coords.max(axis=0)

        try:
            mirror_normals = _estimate_normals_for_pointcloud(
                mirror_pts, sample_stride=1, k=16
            )
        except Exception:
            mirror_normals = np.tile(np.array([0, 0, 1.0]), (len(mirror_pts), 1))

        fov_h_rad = math.radians(fov_h)
        fov_v_rad = math.radians(fov_v)

        default_up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(default_up, proj_dir)) > 0.99:
            default_up = np.array([0.0, 1.0, 0.0])
        right = _normalize(np.cross(proj_dir, default_up))
        up = _normalize(np.cross(right, proj_dir))

        map_x = np.zeros((h_out, w_out), dtype=np.float32)
        map_y = np.zeros((h_out, w_out), dtype=np.float32)

        for yy in range(h_out):
            v = ((yy + 0.5) / h_out - 0.5) * fov_v_rad
            for xx in range(w_out):
                u = ((xx + 0.5) / w_out - 0.5) * fov_h_rad
                dir_cam = _normalize(
                    proj_dir + right * math.tan(u) + up * math.tan(v)
                )

                mirror_hit_pt, _, _ = _nearest_along_ray(
                    proj_origin, dir_cam, mirror_pts, 0.01, 10.0
                )
                if mirror_hit_pt is None:
                    continue

                diffs = mirror_pts - mirror_hit_pt[None, :]
                idx = int(np.argmin(np.linalg.norm(diffs, axis=1)))
                n = mirror_normals[idx]
                if np.linalg.norm(n) == 0:
                    n = screen_normal

                refl = _reflect(dir_cam, n)

                screen_hit_pt, _, _ = _nearest_along_ray(
                    mirror_hit_pt + refl * 1e-6, refl, screen_pts, 0.01, 10.0
                )
                if screen_hit_pt is None:
                    continue

                rel = screen_hit_pt - screen_centroid
                ucoord = float(np.dot(rel, screen_u))
                vcoord = float(np.dot(rel, screen_v))

                if (u_max - u_min) == 0 or (v_max - v_min) == 0:
                    continue

                fx = (ucoord - u_min) / (u_max - u_min)
                fy = (vcoord - v_min) / (v_max - v_min)

                if 0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0:
                    map_x[yy, xx] = fx * (proj_resolution[0] - 1)
                    map_y[yy, xx] = (1.0 - fy) * (proj_resolution[1] - 1)

        warp_cache[cache_key] = (map_x, map_y)
        _log(f"[OK] warp_map prepared for {display_name} ({w_out}x{h_out})", log_func)
        return map_x, map_y

    return None


# --- convert_maps_to_uv_texture_data: OpenGL用UVマップ変換 ---
def convert_maps_to_uv_texture_data(map_x: np.ndarray, map_y: np.ndarray, width: int, height: int) -> bytes:
    """
    OpenCV形式のピクセル単位の remap マップ (map_x, map_y) を、
    ModernGLのテクスチャとして直接使用できるUV座標データに変換する。
    
    Args:
        map_x (numpy.ndarray): X座標マップ (ピクセル単位)
        map_y (numpy.ndarray): Y座標マップ (ピクセル単位)
        width (int): 出力幅
        height (int): 出力高さ

    Returns:
        bytes: ModernGLに転送可能な (H, W, 2) の float32 形式のバイトデータ。
               チャンネル0: U (X座標の正規化値), チャンネル1: V (Y座標の正規化値)
    """
    # 1. 正規化 (ピクセル座標 0～(W-1) / 0～(H-1) を UV座標 0.0～1.0 に変換)
    u = map_x.astype(np.float32) / width
    v = map_y.astype(np.float32) / height
    
    # 2. U (X) と V (Y) のマップをスタックし、(Height, Width, 2) の構造にする
    # シェーダーで `texture(warp_map_tex, v_text).rg` としてアクセスできるようにする
    uv_map = np.dstack((u, v))
    
    # 3. float32 ('f4') のバイト列にして返す
    return uv_map.astype('f4').tobytes()