# grid_utils.py
# グリッド初期化や共通ユーティリティ関数を定義

import math

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

def generate_quad_points(center, normal, width=0.8, height=0.6):
    """
    3D空間上で中心と法線ベクトルからスクリーンの4隅を計算（2D座標に射影）
    """
    n = normalize(normal)
    up = [0, 0, 1] if abs(n[2]) < 0.9 else [0, 1, 0]
    x_axis = normalize(cross(up, n))
    y_axis = normalize(cross(n, x_axis))

    corners = []
    for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        px = [center[i] + dx * width/2 * x_axis[i] + dy * height/2 * y_axis[i] for i in range(3)]
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
    return [x / mag for x in v] if mag > 0 else v
