# grid_utils.py（統合・最新版、モード対応・整合性あり）

import os
import re
import json
import math

from PyQt5.QtGui import QGuiApplication
from config.environment_config import environment

# -----------------------------
# 設定
# -----------------------------
SETTINGS_DIR = "settings"
os.makedirs(SETTINGS_DIR, exist_ok=True)

# -----------------------------
# ファイル関連ユーティリティ
# -----------------------------
def sanitize_filename(name):
    return re.sub(r'[\\/:*?"<>|]', '_', name)

def get_point_path(display_name, mode="perspective"):
    """
    ディスプレイ名と補正モードからポイントファイルのパスを返す。
    """
    safe_name = sanitize_filename(display_name)
    return os.path.join(SETTINGS_DIR, f"{safe_name}_{mode}_points.json")

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
    path = os.path.join(SETTINGS_DIR, "edit_profile.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f).get("display")
    return None

# -----------------------------
# グリッド生成系
# -----------------------------
def generate_perimeter_points(w, h, div=10):
    """
    ディスプレイの外周に沿ったグリッド点を生成。
    """
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
    """
    射影変換用の長方形4点を生成。
    左上→右上→右下→左下
    """
    return [[0, 0], [w, 0], [w, h], [0, h]]

def generate_quad_points(center, normal, width=1.2, height=0.9):
    """
    スクリーンの中心座標・法線から2D画面上の四隅を計算（未スケール）
    """
    n = normalize(normal)
    up = [0, 0, 1] if abs(n[2]) < 0.9 else [0, 1, 0]
    x_axis = normalize(cross(up, n))
    y_axis = normalize(cross(n, x_axis))
    corners = []
    for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        px = [center[i] + dx * width / 2 * x_axis[i] + dy * height / 2 * y_axis[i] for i in range(3)]
        corners.append(px[:2])
    return corners

def auto_generate_from_environment(mode="perspective"):
    """
    environment_config.py に基づいて各画面に初期グリッドを生成・保存（接続状況に応じて調整）
    """
    app = QGuiApplication.instance() or QGuiApplication([])
    screens = QGuiApplication.screens()
    edit_display = load_edit_profile()
    screen_map = {i: s for i, s in enumerate(screens) if s.name() != edit_display}

    screen_defs_all = environment["screens"]
    screen_defs = screen_defs_all[:len(screen_map)]  # 実際の接続数に合わせて切り取る

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
            quad = generate_quad_points(
                screen_def["center"],
                screen_def["normal"],
                width=screen_def.get("width", 1.2),
                height=screen_def.get("height", 0.9)
            )
            points = [[(x + 1) * w / 2, (y + 1) * h / 2] for x, y in quad]

        save_points(name, points, mode=mode)
        print(f"✔ グリッド生成: {name} → {len(points)}点（モード: {mode}）")

    print("🎉 全ディスプレイのグリッド生成完了")


# -----------------------------
# ベクトル演算
# -----------------------------
def cross(a, b):
    return [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ]

def normalize(v):
    mag = math.sqrt(sum(x**2 for x in v))
    return [x / mag for x in v] if mag > 0 else v
