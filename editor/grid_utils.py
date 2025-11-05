import os
import json
import re
import numpy as np
from datetime import datetime
from PyQt5.QtGui import QGuiApplication

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROFILE_DIR = os.path.join(ROOT_DIR, "config", "projector_profiles")
os.makedirs(PROFILE_DIR, exist_ok=True)

def sanitize_filename(display_name: str, mode: str):
    safe_name = re.sub(r'[\\/:*?"<>|.\s]+', "_", display_name)
    safe_name = re.sub(r"^_+", "", safe_name)
    if not safe_name.startswith("__._"):
        safe_name = "__._" + safe_name
    safe_name = re.sub(r"(__\._)+", "__._", safe_name)
    return f"{safe_name}_{mode}_points.json"

def log(msg: str):
    print(f"[DEBUG] {msg}")

def get_point_path(display_name: str, mode: str = "perspective") -> str:
    filename = sanitize_filename(display_name, mode)
    return os.path.join(PROFILE_DIR, filename)

def load_edit_profile():
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

def save_points(display_name: str, points: list, mode: str = "perspective"):
    path = get_point_path(display_name, mode)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(points, f, indent=2, ensure_ascii=False)
        log(f"[SAVE] saved points -> {path}")
    except Exception as e:
        log(f"[ERROR] Failed to save points: {e}")

def load_points(display_name: str, mode: str = "perspective"):
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


# === グリッド生成（画面中央10%サイズ）===
def generate_grid_points(display_name: str, cols: int = 10, rows: int = 10) -> list:
    app = QGuiApplication.instance() or QGuiApplication([])
    screen = next((s for s in QGuiApplication.screens() if s.name() == display_name), None)

    if screen:
        geo = screen.geometry()
        w, h = geo.width(), geo.height()
    else:
        w, h = 1920, 1080

    scale = 0.10
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

    log(f"[INIT_GRID] Display={display_name}  CenterGrid=({left:.1f},{top:.1f})~({right:.1f},{bottom:.1f})")
    return points


def generate_perspective_points(display_name: str) -> list:
    app = QGuiApplication.instance() or QGuiApplication([])
    screen = next((s for s in QGuiApplication.screens() if s.name() == display_name), None)

    if screen:
        geo = screen.geometry()
        w, h = geo.width(), geo.height()
    else:
        w, h = 1920, 1080

    scale = 0.10
    cx, cy = w / 2, h / 2
    half_w, half_h = (w * scale) / 2, (h * scale) / 2

    points = [
        [cx - half_w, cy - half_h],
        [cx + half_w, cy - half_h],
        [cx + half_w, cy + half_h],
        [cx - half_w, cy + half_h],
    ]

    log(f"[INIT_4PT] Display={display_name}  Rect=({cx-half_w:.1f},{cy-half_h:.1f})~({cx+half_w:.1f},{cy+half_h:.1f})")
    return points


# === 修正版：既存ファイルがあれば上書きしない ===
def create_display_grid(display_name: str, mode: str = "warp_map"):
    existing_points = load_points(display_name, mode)
    if existing_points:
        log(f"[SKIP_INIT] 既存グリッドを再利用: {display_name} ({len(existing_points)}点)")
        return existing_points  # ← 再生成せず既存を返す

    # ファイルがなければ初期生成
    if mode == "warp_map":
        points = generate_grid_points(display_name)
    else:
        points = generate_perspective_points(display_name)
    save_points(display_name, points, mode)
    log(f"✔ グリッド生成: {display_name} → {len(points)}点（モード: {mode}）")
    return points


def cleanup_old_profiles():
    for f in os.listdir(PROFILE_DIR):
        if not f.endswith(".json"):
            continue
        old_path = os.path.join(PROFILE_DIR, f)
        if f.startswith("__.___._"):
            fixed = f.replace("__.___._", "__._", 1)
            os.rename(old_path, os.path.join(PROFILE_DIR, fixed))
            log(f"[CLEANUP] renamed: {f} → {fixed}")
    log("🧹 古いファイル名のクリーニング完了")

def auto_generate_from_environment(mode="warp_map", displays=None):
    app = QGuiApplication.instance() or QGuiApplication([])
    screens = QGuiApplication.screens()
    if not displays:
        primary = QGuiApplication.primaryScreen().name()
        displays = [s.name() for s in screens if s.name() != primary]
    cleanup_old_profiles()
    for name in displays:
        create_display_grid(name, mode)
    print(f"🎉 選択ディスプレイ（{len(displays)}台）のグリッドを生成完了。")
