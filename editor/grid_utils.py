import os
import json
import re
import numpy as np
from datetime import datetime
from PyQt5.QtGui import QGuiApplication

# === 定数 ===
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROFILE_DIR = os.path.join(ROOT_DIR, "config", "projector_profiles")
os.makedirs(PROFILE_DIR, exist_ok=True)

# === 基本ユーティリティ ===
def sanitize_filename(display_name: str, mode: str):
    """ディスプレイ名とモードに基づいて安全で一意なファイル名を作成"""
    # すべての特殊文字をアンダースコアに
    safe_name = re.sub(r'[\\/:*?"<>|.\s]+', "_", display_name)

    # 既存の "__._" があれば重複を防ぐ
    safe_name = re.sub(r"^_+", "", safe_name)  # 先頭の "_" 群を削除
    if not safe_name.startswith("__._"):
        safe_name = "__._" + safe_name

    # プレフィックスの重複をさらに一段階防止
    safe_name = re.sub(r"(__\._)+", "__._", safe_name)

    return f"{safe_name}_{mode}_points.json"


def log(msg: str):
    print(f"[DEBUG] {msg}")

# === ファイル名生成 ===
def get_point_path(display_name: str, mode: str = "perspective") -> str:
    """指定ディスプレイの補正点保存パスを返す"""
    filename = sanitize_filename(display_name, mode)
    return os.path.join(PROFILE_DIR, filename)

# === 編集プロファイル読み込み ===
def load_edit_profile():
    """編集プロファイル(edit_profile.json)を読み込む"""
    profile_path = os.path.join(os.path.dirname(__file__), "..", "config", "edit_profile.json")
    profile_path = os.path.abspath(profile_path)

    if not os.path.exists(profile_path):
        print(f"[WARN] edit_profile.json が見つかりません: {profile_path}")
        return {}

    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[DEBUG] 編集プロファイル読込成功: {profile_path}")
        return data
    except Exception as e:
        print(f"[ERROR] 編集プロファイル読込失敗: {e}")
        return {}

# === 読み込み / 保存 ===
def save_points(display_name: str, points: list, mode: str = "perspective"):
    """ディスプレイごとの補正点を保存"""
    path = get_point_path(display_name, mode)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(points, f, indent=2, ensure_ascii=False)
        log(f"[SAVE] saved points -> {path}")
    except Exception as e:
        log(f"[ERROR] Failed to save points: {e}")

def load_points(display_name: str, mode: str = "perspective"):
    """保存済みの補正点を読み込む"""
    path = get_point_path(display_name, mode)
    if not os.path.exists(path):
        log(f"[DEBUG] グリッドファイルが存在しません: {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        log(f"[LOAD] loaded points <- {path}")
        return data
    except Exception as e:
        log(f"[ERROR] Failed to load points: {e}")
        return None

# === グリッド生成 ===
def generate_grid_points(display_name: str, cols: int = 10, rows: int = 10) -> list:
    """
    ディスプレイ全体をカバーする均等グリッドを生成。
    出力は画面ピクセル単位。
    """
    app = QGuiApplication.instance() or QGuiApplication([])
    screen = next((s for s in QGuiApplication.screens() if s.name() == display_name), None)

    if screen:
        geo = screen.geometry()
        w, h = geo.width(), geo.height()
    else:
        w, h = 1920, 1080  # fallback

    points = []
    for y in range(rows):
        for x in range(cols):
            px = x / (cols - 1) * w
            py = y / (rows - 1) * h
            points.append([px, py])
    return points


def generate_perspective_points(display_name: str) -> list:
    """
    perspective（斜影変換）モードの初期4点を画面端に配置。
    """
    app = QGuiApplication.instance() or QGuiApplication([])
    screen = next((s for s in QGuiApplication.screens() if s.name() == display_name), None)

    if screen:
        geo = screen.geometry()
        w, h = geo.width(), geo.height()
    else:
        w, h = 1920, 1080

    return [
        [0, 0],        # 左上
        [w, 0],        # 右上
        [w, h],        # 右下
        [0, h],        # 左下
    ]


def create_display_grid(display_name: str, mode: str = "warp_map"):
    """モード別にグリッドを生成して保存"""
    if mode == "warp_map":
        # 外周のみ（縦横10点分割）
        app = QGuiApplication.instance() or QGuiApplication([])
        screen = next((s for s in QGuiApplication.screens() if s.name() == display_name), None)
        if screen:
            geo = screen.geometry()
            w, h = geo.width(), geo.height()
        else:
            w, h = 1920, 1080

        points = generate_perimeter_points(w, h, div=10)  # ← 外周のみ生成

    elif mode == "perspective":
        points = generate_perspective_points(display_name)

    else:
        # その他（保険として）
        app = QGuiApplication.instance() or QGuiApplication([])
        screen = next((s for s in QGuiApplication.screens() if s.name() == display_name), None)
        if screen:
            geo = screen.geometry()
            w, h = geo.width(), geo.height()
        else:
            w, h = 1920, 1080
        points = generate_perimeter_points(w, h, div=10)

    save_points(display_name, points, mode)
    log(f"✔ グリッド生成: {display_name} → {len(points)}点（モード: {mode}）")
    return points


# === 全ディスプレイ一括生成 ===
def generate_all_displays_grid(displays: list):
    """複数ディスプレイに対してグリッドを自動生成"""
    for d in displays:
        create_display_grid(d, "warp_map")
    log("🎉 全ディスプレイのグリッド生成完了")

def generate_perimeter_points(w, h, div=10):
    points = []
    for i in range(div):
        points.append([w * i / (div - 1), 0])
    for i in range(1, div - 1):
        points.append([w, h * i / (div - 1)])
    for i in reversed(range(div)):
        points.append([w * i / (div - 1), h])
    for i in reversed(range(1, div - 1)):
        points.append([0, h * i / (div - 1)])
    return points


# === 旧ファイル検出・整理 ===
def list_existing_profiles():
    """保存済みの全補正ファイルを一覧表示"""
    files = [f for f in os.listdir(PROFILE_DIR) if f.endswith(".json")]
    return sorted(files)

def cleanup_old_profiles():
    """旧フォーマットのファイルをリネームして統一"""
    for f in list_existing_profiles():
        old_path = os.path.join(PROFILE_DIR, f)
        if f.startswith("__.___._"):  # 二重接頭辞を検出
            fixed_name = f.replace("__.___._", "__._", 1)
            new_path = os.path.join(PROFILE_DIR, fixed_name)
            os.rename(old_path, new_path)
            log(f"[CLEANUP] renamed: {f} → {fixed_name}")
    log("🧹 古いファイル名のクリーニング完了")

def auto_generate_from_environment(mode="warp_map", displays=None):
    """
    現在の接続ディスプレイ情報から、選択された or 全ディスプレイ分の
    グリッドJSONを自動生成する。
    """
    app = QGuiApplication.instance() or QGuiApplication([])
    screens = QGuiApplication.screens()

    # 指定がなければ、編集ディスプレイ（DISPLAY1など）以外を全対象とする
    if not displays:
        primary = QGuiApplication.primaryScreen().name()
        displays = [s.name() for s in screens if s.name() != primary]

    if not displays:
        print("[WARN] グリッドを生成するディスプレイが見つかりません。")
        return

    cleanup_old_profiles()  # 重複プレフィックス修正
    for name in displays:
        create_display_grid(name, mode)
    print(f"🎉 選択ディスプレイ（{len(displays)}台）のグリッドを生成完了。")

# === 凸面鏡ワープマップ生成 ===
def generate_mirror_warp_map(projector, mirror, screen, resolution=(1920, 1080),
                             mirror_radius=0.2475, screen_radius=2.204, screen_center_height=1.650):
    """
    凸面鏡反射を考慮してプロジェクター→スクリーン間のwarp_mapを生成する。
    - projector, mirror, screen: dict {"position": [x,y,z], "forward": [x,y,z]}
    - resolution: 出力画像の解像度 (width, height)
    - mirror_radius: 凸面鏡半径 [m]（例: 直径495mmの1/4球 → 半径0.2475m）
    - screen_radius: スクリーン半径 [m]
    - screen_center_height: スクリーン中心高さ [m]

    戻り値:
        map_x, map_y : np.float32
    """

    width, height = resolution
    px, py, pz = projector["position"]
    mx, my, mz = mirror["position"]
    sx, sy, sz = screen["position"]

    # 座標系: スクリーン中心を原点、Z方向が前方（視線方向）
    # 画素グリッドを生成（プロジェクター側から見た像）
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    xv, yv = np.meshgrid(x, y)

    # プロジェクター空間上の仮想視線ベクトル
    rays = np.stack([xv, -yv, np.ones_like(xv)], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)

    # プロジェクター位置ベクトル
    proj_pos = np.array([px, py, pz])
    mirror_pos = np.array([mx, my, mz])

    # 鏡面の法線方向を設定（Z軸向き）
    mirror_normal = np.array([0, 0, -1])

    # 凸面鏡の反射点を近似的に計算
    # → 投影ベクトルを鏡面上に伸ばして反射方向を求める
    t = np.dot(mirror_normal, mirror_pos - proj_pos) / np.dot(mirror_normal, rays)
    hit_points = proj_pos + rays * t[..., np.newaxis]

    # 鏡面上の法線（球の中心を原点とした法線ベクトル）
    mirror_center = mirror_pos - mirror_normal * mirror_radius
    normal_vecs = hit_points - mirror_center
    normal_vecs /= np.linalg.norm(normal_vecs, axis=-1, keepdims=True)

    # 反射ベクトル
    reflect_rays = rays - 2 * np.sum(rays * normal_vecs, axis=-1, keepdims=True) * normal_vecs

    # 反射後にスクリーン（Z=0 近辺）と交差する点を求める
    # ここではスクリーンを球面（半径 screen_radius）として近似
    screen_center = np.array([0, 0, screen_center_height])
    A = np.sum(reflect_rays**2, axis=-1)
    B = 2 * np.sum((hit_points - screen_center) * reflect_rays, axis=-1)
    C = np.sum((hit_points - screen_center)**2, axis=-1) - screen_radius**2

    # 二次方程式を解いて交点距離 t2 を求める
    discriminant = B**2 - 4 * A * C
    t2 = np.where(discriminant > 0, (-B + np.sqrt(discriminant)) / (2 * A), np.nan)
    screen_points = hit_points + reflect_rays * t2[..., np.newaxis]

    # 交点をスクリーン平面上の座標に射影
    # ここではθφ座標（緯度経度）に変換して正規化する
    rel = screen_points - screen_center
    theta = np.arctan2(rel[..., 0], rel[..., 2])  # 横方向角度
    phi = np.arctan2(rel[..., 1], np.sqrt(rel[..., 0]**2 + rel[..., 2]**2))  # 縦方向角度

    # θφをピクセル座標にマッピング
    map_x = (theta - theta.min()) / (theta.max() - theta.min()) * width
    map_y = (phi - phi.min()) / (phi.max() - phi.min()) * height

    map_x = np.nan_to_num(map_x).astype(np.float32)
    map_y = np.nan_to_num(map_y).astype(np.float32)

    return map_x, map_y

# === 動作テスト ===
if __name__ == "__main__":
    displays = ["\\\\.\\DISPLAY1", "\\\\.\\DISPLAY2"]
    cleanup_old_profiles()
    generate_all_displays_grid(displays)
