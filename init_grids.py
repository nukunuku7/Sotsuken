# init_grids.py（環境定義に基づくモード別グリッド自動生成対応）
import os
import json
import re
import math
from PyQt5.QtGui import QGuiApplication
from settings.config.environment_config import environment
from grid_utils import generate_perimeter_points, generate_perspective_points

SETTINGS_DIR = "C:/Users/vrlab/.vscode/nukunuku/Sotsuken/settings"
POINT_SUFFIX = "_points.json"


def sanitize_filename(name):
    return re.sub(r'[\\/:*?"<>|]', '_', name)


def save_points(display_name, points):
    path = os.path.join(SETTINGS_DIR, f"{sanitize_filename(display_name)}{POINT_SUFFIX}")
    with open(path, "w") as f:
        json.dump(points, f)


def load_edit_profile():
    path = os.path.join(SETTINGS_DIR, "edit_profile.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f).get("display")
    return None


def generate_quad_points(center, normal, width=0.8, height=0.6):
    n = normalize(normal)
    up = [0, 0, 1] if abs(n[2]) < 0.9 else [0, 1, 0]
    x_axis = normalize(cross(up, n))
    y_axis = normalize(cross(n, x_axis))
    corners = []
    for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        px = [center[i] + dx * width / 2 * x_axis[i] + dy * height / 2 * y_axis[i] for i in range(3)]
        corners.append(px[:2])
    return corners


def cross(a, b):
    return [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ]


def normalize(v):
    mag = math.sqrt(sum(x**2 for x in v))
    return [x / mag for x in v]


def auto_generate_from_environment(mode="perspective"):
    app = QGuiApplication([])
    screens = QGuiApplication.screens()
    edit_display = load_edit_profile()
    screen_map = {i: s for i, s in enumerate(screens) if s.name() != edit_display}

    screen_defs = environment["screens"]
    if len(screen_defs) > len(screen_map):
        print("[警告] スクリーン数が接続ディスプレイより多いです")

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

        save_points(name, points)
        print(f"✔ グリッド生成: {name} → {len(points)}点（モード: {mode}）")

    print("🎉 全ディスプレイのグリッド生成完了")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["perspective", "warp_map"], default="perspective")
    args = parser.parse_args()
    auto_generate_from_environment(mode=args.mode)
