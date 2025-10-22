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
    画面中央20%サイズに寄せた均等グリッドを生成。
    出力は画面ピクセル単位。
    """
    app = QGuiApplication.instance() or QGuiApplication([])
    screen = next((s for s in QGuiApplication.screens() if s.name() == display_name), None)

    if screen:
        geo = screen.geometry()
        w, h = geo.width(), geo.height()
    else:
        w, h = 1920, 1080  # fallback

    scale = 0.2
    cx, cy = w / 2, h / 2
    half_w, half_h = (w * scale) / 2, (h * scale) / 2
    left, right = cx - half_w, cx + half_w
    top, bottom = cy - half_h, cy + half_h

    points = []
    for j in range(rows):
        y = top + (bottom - top) * (j / (rows - 1))
        for i in range(cols):
            x = left + (right - left) * (i / (cols - 1))
            points.append([x, y])
    return points



def generate_perspective_points(display_name: str) -> list:
    """
    perspective（斜影変換）モードの初期4点を画面中央20%に寄せて配置。
    """
    app = QGuiApplication.instance() or QGuiApplication([])
    screen = next((s for s in QGuiApplication.screens() if s.name() == display_name), None)

    if screen:
        geo = screen.geometry()
        w, h = geo.width(), geo.height()
    else:
        w, h = 1920, 1080

    scale = 0.1
    cx, cy = w / 2, h / 2
    half_w, half_h = (w * scale) / 2, (h * scale) / 2

    return [
        [cx - half_w, cy - half_h],  # 左上
        [cx + half_w, cy - half_h],  # 右上
        [cx + half_w, cy + half_h],  # 右下
        [cx - half_w, cy + half_h],  # 左下
    ]


def create_display_grid(display_name: str, mode: str = "warp_map"):
    """モード別にグリッドを生成して保存（重複した内部関数を削除して整理）"""
    app = QGuiApplication.instance() or QGuiApplication([])
    screen = next((s for s in QGuiApplication.screens() if s.name() == display_name), None)
    if screen:
        geo = screen.geometry()
        w, h = geo.width(), geo.height()
    else:
        w, h = 1920, 1080

    if mode == "warp_map":
        # 画面中心寄りの外周（margin_ratio内側）に10分割点を生成
        margin_ratio = 0.1
        margin_x = w * margin_ratio
        margin_y = h * margin_ratio
        left, right = margin_x, w - margin_x
        top, bottom = margin_y, h - margin_y
        inner_w, inner_h = right - left, bottom - top

        # グローバルの generate_perimeter_points を再利用し、オフセットを加える
        raw = generate_perimeter_points(inner_w, inner_h, div=10)
        points = [[x + left, y + top] for x, y in raw]
    elif mode == "perspective":
        points = generate_perspective_points(display_name)
    else:
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
