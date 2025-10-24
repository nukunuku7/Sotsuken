import os
import json
import glob

def polygon_area(points):
    """2Dポリゴンの面積をShoelace formulaで求める"""
    n = len(points)
    area = 0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2

def load_points_from_json(file_path):
    """グリッドJSONから座標リストを読み込む（list形式にも対応）"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ① list形式（[[x, y], [x, y], ...]）
    if isinstance(data, list) and all(isinstance(p, list) and len(p) == 2 for p in data):
        return [(float(p[0]), float(p[1])) for p in data]

    # ② dict形式（{"points": [{"x": ..., "y": ...}, ...]}）
    elif isinstance(data, dict) and "points" in data:
        return [(float(p["x"]), float(p["y"])) for p in data["points"]]

    else:
        print(f"⚠ 未対応のJSON形式: {file_path}")
        return []

def calculate_pixel_usage(points, width=1920, height=1080):
    """ポリゴン面積 / 全ピクセル面積 の割合を計算"""
    if not points:
        return 0
    area = polygon_area(points)
    total = width * height
    return (area / total) * 100

def main():
    print("📊 Pixel Usage Percentage Calculator")
    print("------------------------------------")

    grid_dir = os.path.join(os.path.dirname(__file__), "config", "projector_profiles")
    json_files = glob.glob(os.path.join(grid_dir, "*.json"))

    if not json_files:
        print("⚠ グリッドJSONが見つかりませんでした。")
        return

    for file_path in json_files:
        points = load_points_from_json(file_path)
        usage = calculate_pixel_usage(points)
        print(f"{os.path.basename(file_path):<40} → {usage:.2f}% pixel usage")

if __name__ == "__main__":
    main()
