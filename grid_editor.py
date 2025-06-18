# grid_editor.py（10x10外周点対応 + 緑線 + 黒マスク + JSON保存 + リアルタイム反映）
import argparse
import tkinter as tk
import json
import os
import re

SETTINGS_DIR = "C:/Users/vrlab/.vscode/nukunuku/Sotsuken/settings"
os.makedirs(SETTINGS_DIR, exist_ok=True)

POINT_RADIUS = 8
GRID_DIV = 10  # 点数10分割


def sanitize_filename(name):
    return re.sub(r'[\\/:*?"<>|]', '_', name)


def generate_perimeter_points(w, h, div):
    points = []
    for i in range(div):  # 上辺
        points.append([w * i / (div - 1), 0])
    for i in range(1, div - 1):  # 右辺
        points.append([w, h * i / (div - 1)])
    for i in reversed(range(div)):  # 下辺
        points.append([w * i / (div - 1), h])
    for i in reversed(range(1, div - 1)):  # 左辺
        points.append([0, h * i / (div - 1)])
    return points


class EditorCanvas(tk.Canvas):
    def __init__(self, master, args, **kwargs):
        super().__init__(master, **kwargs)
        self.args = args
        self.points = self.load_points()
        self.dragging_point = None
        self.bind("<ButtonPress-1>", self.on_press)
        self.bind("<B1-Motion>", self.on_drag)
        self.bind("<ButtonRelease-1>", self.on_release)
        self.draw()

    def load_points(self):
        path = os.path.join(SETTINGS_DIR, f"{sanitize_filename(self.args.display)}_points.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return generate_perimeter_points(self.args.w, self.args.h, GRID_DIV)

    def save_points(self):
        path = os.path.join(SETTINGS_DIR, f"{sanitize_filename(self.args.display)}_points.json")
        with open(path, "w") as f:
            json.dump(self.points, f)

    def draw(self):
        self.delete("all")

        # 黒マスク
        self.create_rectangle(0, 0, self.args.w, self.args.h, fill="black")

        # 外周内側の透過マスク
        self.create_polygon(*sum(self.points, []), fill="gray20")

        # 緑線（外周）
        for i in range(len(self.points)):
            x1, y1 = self.points[i]
            x2, y2 = self.points[(i + 1) % len(self.points)]
            self.create_line(x1, y1, x2, y2, fill="green", width=2)

        # 赤い点
        for x, y in self.points:
            self.create_oval(
                x - POINT_RADIUS, y - POINT_RADIUS,
                x + POINT_RADIUS, y + POINT_RADIUS,
                fill="red")

    def on_press(self, event):
        for i, (x, y) in enumerate(self.points):
            if abs(event.x - x) < POINT_RADIUS and abs(event.y - y) < POINT_RADIUS:
                self.dragging_point = i
                return

    def on_drag(self, event):
        if self.dragging_point is not None:
            self.points[self.dragging_point] = [event.x, event.y]
            self.save_points()
            self.draw()

    def on_release(self, event):
        self.dragging_point = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['editor', 'projector'], required=True)
    parser.add_argument('--display', type=str, required=True)
    parser.add_argument('--x', type=int, required=True)
    parser.add_argument('--y', type=int, required=True)
    parser.add_argument('--w', type=int, required=True)
    parser.add_argument('--h', type=int, required=True)
    return parser.parse_args()


def setup_window(args):
    root = tk.Tk()
    root.title(f"{args.display} - {args.mode}")
    root.geometry(f"{args.w}x{args.h}+{args.x}+{args.y}")
    root.attributes('-fullscreen', True)
    return root


def main():
    args = parse_args()
    root = setup_window(args)

    if args.mode == 'editor':
        canvas = EditorCanvas(root, args, width=args.w, height=args.h, bg="black")
        canvas.pack(fill="both", expand=True)
    else:
        # projectorモード未使用（編集モードのみ対応）
        pass

    root.mainloop()


if __name__ == "__main__":
    main()
