import os
import re
import json
import math
from PyQt5.QtGui import QGuiApplication
from config.environment_config import environment

# -----------------------------
# 設定
# -----------------------------
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")
PROFILE_DIR = os.path.join(CONFIG_DIR, "projector_profiles")
os.makedirs(PROFILE_DIR, exist_ok=True)

# -----------------------------
# ファイル関連ユーティリティ
# -----------------------------
def sanitize_filename(name):
    return re.sub(r'[\\/:*?"<>|]', '_', name)

def get_point_path(display_name, mode="perspective"):
    safe_name = sanitize_filename(display_name)
    return os.path.join(PROFILE_DIR, f"{safe_name}_{mode}_points.json")

def save_points(display_name, points, mode="perspective"):
    path = get_point_path(display_name, mode)
    with open(path, "w") as f:
        json.dump(points, f)

def load_points(display_name, mode="perspective"):
    path = get_point_path(display_name, mode)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

def load_edit_profile():
    path = os.path.join(CONFIG_DIR, "edit_profile.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f).get("display")
    return None

# -----------------------------
# グリッド生成系
# -----------------------------
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

def generate_grid_from_quad(corners, div=10):
    if len(corners) != 4:
        raise ValueError(f"quadの定義は4点必要ですが {len(corners)} 点しかありません: {corners}")
    p00, p10, p11, p01 = corners
    grid = []
    for i in range(div + 1):
        u = i / div
        left = [p00[j] * (1 - u) + p01[j] * u for j in range(2)]
        right = [p10[j] * (1 - u) + p11[j] * u for j in range(2)]
        for j in range(div + 1):
            v = j / div
            pt = [left[k] * (1 - v) + right[k] * v for k in range(2)]
            grid.append(pt)
    return grid

def auto_generate_from_environment(mode="perspective"):
    app = QGuiApplication.instance() or QGuiApplication([])
    screens = QGuiApplication.screens()
    edit_display = load_edit_profile()
    screen_map = {i: s for i, s in enumerate(screens) if s.name() != edit_display}

    screen_defs_all = environment["screens"]
    screen_defs = screen_defs_all[:len(screen_map)]

    if len(screen_defs_all) > len(screen_map):
        print("[警告] 定義されたスクリーン数が接続ディスプレイより多いため、一部は省略されます。")
    elif len(screen_defs_all) < len(screen_map):
        print("[警告] 接続ディスプレイの数が定義より多いため、余剰ディスプレイは無視されます。")

    for (i, screen), screen_def in zip(screen_map.items(), screen_defs):
        name = screen.name()
        geom = screen.geometry()
        w, h = geom.width(), geom.height()

        if mode == "warp_map":
            points = generate_perimeter_points(w, h, div=10)
        else:
            # perspectiveモード: 仮の矩形
            points = generate_perspective_points(w, h)

        save_points(name, points, mode=mode)
        print(f"✔ グリッド生成: {name} → {len(points)}点（モード: {mode}）")

    print("🎉 全ディスプレイのグリッド生成完了")

# -----------------------------
# ベクトル演算
# -----------------------------
def cross(a, b):
    return [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]

def normalize(v):
    mag = math.sqrt(sum(x**2 for x in v))
    return [x / mag for x in v] if mag > 0 else v
