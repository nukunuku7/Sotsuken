# grid_editor.py（ズーム・パン・自動補正初期グリッド連携対応＋モード別初期グリッド）
import argparse
import tkinter as tk
import json
import os
import re
import math

SETTINGS_DIR = "C:/Users/vrlab/.vscode/nukunuku/Sotsuken/settings"
os.makedirs(SETTINGS_DIR, exist_ok=True)

POINT_RADIUS = 8
GRID_DIV = 10

def sanitize_filename(name):
    return re.sub(r'[\/:*?"<>|]', '_', name)

def generate_perimeter_points(w, h, div):
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

def get_point_path(display_name):
    return os.path.join(SETTINGS_DIR, f"{sanitize_filename(display_name)}_points.json")

def generate_quad(w, h):
    margin = 0.05
    return [
        [w * margin, h * margin],
        [w * (1 - margin), h * margin],
        [w * (1 - margin), h * (1 - margin)],
        [w * margin, h * (1 - margin)]
    ]

def load_or_generate_points(display_name, w, h, mode):
    path = get_point_path(display_name)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    else:
        if mode == "perspective":
            return generate_quad(w, h)
        else:
            return generate_quad(w, h, GRID_DIV)

def save_points(display_name, points):
    path = get_point_path(display_name)
    with open(path, "w") as f:
        json.dump(points, f)

class EditorCanvas(tk.Canvas):
    def __init__(self, master, args, **kwargs):
        super().__init__(master, **kwargs)
        self.args = args
        self.display_name = args.display
        self.points = self.load_initial_points()
        self.dragging_point = None

        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.drag_start = None

        self.bind("<ButtonPress-1>", self.on_press)
        self.bind("<B1-Motion>", self.on_drag)
        self.bind("<ButtonRelease-1>", self.on_release)
        self.bind("<MouseWheel>", self.on_zoom)
        self.bind("<ButtonPress-2>", self.on_pan_start)
        self.bind("<B2-Motion>", self.on_pan_move)

        self.draw()

    def canvas_to_world(self, x, y):
        return (x - self.offset_x) / self.scale, (y - self.offset_y) / self.scale

    def world_to_canvas(self, x, y):
        return x * self.scale + self.offset_x, y * self.scale + self.offset_y

    def draw(self):
        self.delete("all")
        self.create_rectangle(0, 0, self.winfo_width(), self.winfo_height(), fill="black")
        transformed_points = [self.world_to_canvas(x, y) for x, y in self.points]
        self.create_polygon(*sum(transformed_points, ()), fill="gray20")
        for i in range(len(transformed_points)):
            x1, y1 = transformed_points[i]
            x2, y2 = transformed_points[(i + 1) % len(transformed_points)]
            self.create_line(x1, y1, x2, y2, fill="green", width=2)
        for x, y in transformed_points:
            self.create_oval(x - POINT_RADIUS, y - POINT_RADIUS, x + POINT_RADIUS, y + POINT_RADIUS, fill="red")

    def on_press(self, event):
        wx, wy = self.canvas_to_world(event.x, event.y)
        for i, (x, y) in enumerate(self.points):
            if abs(wx - x) < POINT_RADIUS / self.scale and abs(wy - y) < POINT_RADIUS / self.scale:
                self.dragging_point = i
                return

    def on_drag(self, event):
        if self.dragging_point is not None:
            wx, wy = self.canvas_to_world(event.x, event.y)
            self.points[self.dragging_point] = [wx, wy]
            save_points(self.display_name, self.points)
            self.draw()

    def on_release(self, event):
        self.dragging_point = None
        save_points(self.display_name, self.points)
        self.draw()

    def on_zoom(self, event):
        scale_factor = 1.1 if event.delta > 0 else 0.9
        cx, cy = event.x, event.y
        wx, wy = self.canvas_to_world(cx, cy)
        self.scale *= scale_factor
        self.offset_x = cx - wx * self.scale
        self.offset_y = cy - wy * self.scale
        self.draw()

    def on_pan_start(self, event):
        self.drag_start = (event.x, event.y)

    def on_pan_move(self, event):
        if self.drag_start:
            dx = event.x - self.drag_start[0]
            dy = event.y - self.drag_start[1]
            self.offset_x += dx
            self.offset_y += dy
            self.drag_start = (event.x, event.y)
            self.draw()
    
    def load_initial_points(self):
        path = get_point_path(self.display_name)
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        if self.args.mode == "perspective":
            return generate_quad(self.args.w, self.args.h)
        else:
            return generate_perimeter_points(self.args.w, self.args.h, GRID_DIV)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['perspective', 'warp_map'], required=True)
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
    return root


def main():
    args = parse_args()
    root = setup_window(args)
    canvas = EditorCanvas(root, args, width=args.w, height=args.h, bg="black")
    canvas.pack(fill="both", expand=True)
    root.mainloop()


if __name__ == "__main__":
    main()
