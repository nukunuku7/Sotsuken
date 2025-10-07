import os
import json
import re
from datetime import datetime
from PyQt5.QtGui import QGuiApplication

# === 定数 ===
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILE_DIR = os.path.join(ROOT_DIR, "config", "projector_profiles")
os.makedirs(PROFILE_DIR, exist_ok=True)

# === 基本ユーティリティ ===
def sanitize_filename(name: str) -> str:
    """Windowsでも安全に扱えるファイル名に変換"""
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = name.replace(" ", "_")
    return name.strip("_")

def log(msg: str):
    print(f"[DEBUG] {msg}")

# === ファイル名生成 ===
def get_point_path(display_name: str, mode: str = "perspective") -> str:
    """
    指定ディスプレイの補正点保存パスを返す。
    "__._" プレフィックスが重複しないよう自動判定。
    """
    base = display_name.replace("\\", "_").replace(":", "_")
    safe_name = sanitize_filename(base)

    # "__._" がすでに含まれている場合は重複回避
    if not safe_name.startswith("__._"):
        safe_name = "__._" + safe_name

    filename = f"{safe_name}_{mode}_points.json"
    return os.path.join(PROFILE_DIR, filename)

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
def generate_grid_points(cols: int = 6, rows: int = 6) -> list:
    """cols×rows の2Dグリッド点を生成"""
    return [[x / (cols - 1), y / (rows - 1)] for y in range(rows) for x in range(cols)]

def create_display_grid(display_name: str, mode: str = "warp_map"):
    """ディスプレイごとに新しいグリッドを生成して保存"""
    points = generate_grid_points()
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


def generate_perspective_points(w, h):
    return [[0, 0], [w, 0], [w, h], [0, h]]

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
