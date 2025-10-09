import os
import json
import re
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


# === 動作テスト ===
if __name__ == "__main__":
    displays = ["\\\\.\\DISPLAY1", "\\\\.\\DISPLAY2"]
    cleanup_old_profiles()
    generate_all_displays_grid(displays)
