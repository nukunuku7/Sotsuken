# generate_environment_360.py

import numpy as np
from scipy.optimize import minimize
import os

# スクリーンとプロジェクション設定
RADIUS = 1102 / 1000  # スクリーン半径 (m)
CENTER_Z = 1650 / 1000  # スクリーン中心高さ (m)
MIRROR_RADIUS = 0.495 / 2  # 鏡の半径 (m)
MIRROR_OFFSET = 0.7  # 鏡までのオフセット距離
NUM_SCREENS = 3
FOV = 100.0  # プロジェクターの水平視野角

CONFIG_PATH = "settings/config/environment_config.py"  # 出力先パス

def rotation_matrix_z(degrees):
    theta = np.radians(degrees)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

def get_screen_info(index):
    angle = 120 * index
    rot = rotation_matrix_z(angle)
    center = rot @ np.array([RADIUS, 0, CENTER_Z])
    normal = rot @ np.array([-1, 0, 0])
    return center.tolist(), normal.tolist()

def optimize_projector(mirror_pos, screen_center):
    def cost(projector_pos):
        projector_pos = np.array(projector_pos)
        v1 = mirror_pos - projector_pos
        v2 = screen_center - mirror_pos
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        return 1 - np.dot(v1, v2)

    init = mirror_pos + np.array([-0.5, 0, 0])
    bounds = [(-2, 2), (-2, 2), (0.2, 2)]
    res = minimize(cost, init, method='SLSQP', bounds=bounds)
    return res.x

def to_python_literal(env_dict):
    """辞書をPythonコード文字列に変換"""
    import pprint
    return "environment = " + pprint.pformat(env_dict, indent=4)

def create_environment_py():
    screens = []
    projectors = []
    mirrors = []

    for i in range(NUM_SCREENS):
        center, normal = get_screen_info(i)
        screens.append({
            "id": f"screen{i+1}",
            "center": center,
            "normal": normal,
            "radius": round(RADIUS, 4)
        })

        direction = np.array(normal)
        mirror_pos = np.array(center) - direction * MIRROR_OFFSET
        mirrors.append({
            "id": f"mirror{i+1}",
            "center": mirror_pos.tolist(),
            "radius": round(MIRROR_RADIUS, 4)
        })

        proj_pos = optimize_projector(mirror_pos, center)
        projectors.append({
            "id": f"proj{i+1}",
            "position": proj_pos.tolist(),
            "fov": FOV
        })

    environment = {
        "screens": screens,
        "projectors": projectors,
        "mirrors": mirrors
    }

    return to_python_literal(environment)

if __name__ == "__main__":
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    config_code = create_environment_py()

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write("# 自動生成された環境構成\n\n")
        f.write(config_code + "\n")

    print(f"✅ {CONFIG_PATH} に書き出しました")
