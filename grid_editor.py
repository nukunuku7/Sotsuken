# grid_editor.py
import argparse
import tkinter as tk
import json
import os

POINT_RADIUS = 10
SAVE_PATH = "./outlines"
os.makedirs(SAVE_PATH, exist_ok=True)

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
        filepath = os.path.join(SAVE_PATH, f"{self.args.display}_points.json")
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                return json.load(f)
        # 初期化（時計回りに四隅）
        w, h = self.args.w, self.args.h
        return [
            [w * 0.2, h * 0.2],
            [w * 0.8, h * 0.2],
            [w * 0.8, h * 0.8],
            [w * 0.2, h * 0.8]
        ]

    def save_points(self):
        filepath = os.path.join(SAVE_PATH, f"{self.args.display}_points.json")
        with open(filepath, "w") as f:
            json.dump(self.points, f)

    def draw(self):
        self.delete("all")
        # 背景マスク
        self.create_rectangle(0, 0, self.args.w, self.args.h, fill="black")

        # 外周マスクの中を透過的に見せる
        self.create_polygon(*sum(self.points, []), fill="gray20")

        # 線（外周）
        for i in range(len(self.points)):
            x1, y1 = self.points[i]
            x2, y2 = self.points[(i + 1) % len(self.points)]
            self.create_line(x1, y1, x2, y2, fill="green", width=2)

        # 点
        for x, y in self.points:
            self.create_oval(
                x - POINT_RADIUS, y - POINT_RADIUS,
                x + POINT_RADIUS, y + POINT_RADIUS,
                fill="red"
            )

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
    parser.add_argument('--mode', type=str, choices=['projector', 'editor'], required=True)
    parser.add_argument('--display', type=str, required=True)
    parser.add_argument('--x', type=int, required=True)
    parser.add_argument('--y', type=int, required=True)
    parser.add_argument('--w', type=int, required=True)
    parser.add_argument('--h', type=int, required=True)
    return parser.parse_args()

def setup_window(args):
    root = tk.Tk()
    root.title(f"{args.mode.capitalize()} - {args.display}")
    root.geometry(f"{args.w}x{args.h}+{args.x}+{args.y}")
    root.attributes('-fullscreen', True)
    return root

def run_projector_mode(canvas, args):
    canvas.create_text(args.w//2, args.h//2, text="Projector Mode", font=("Arial", 50), fill="white")

def run_editor_mode(root, args):
    canvas = EditorCanvas(root, args, width=args.w, height=args.h, bg="black")
    canvas.pack(fill="both", expand=True)

def main():
    args = parse_args()
    root = setup_window(args)

    if args.mode == 'editor':
        run_editor_mode(root, args)
    else:
        canvas = tk.Canvas(root, width=args.w, height=args.h, bg="black")
        canvas.pack(fill="both", expand=True)
        run_projector_mode(canvas, args)

    root.mainloop()

if __name__ == "__main__":
    main()
