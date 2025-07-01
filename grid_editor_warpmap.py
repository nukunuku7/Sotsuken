# grid_editor_warpmap.py（環境設定ベースの初期グリッド対応）

import argparse
import tkinter as tk
import json
import os
from grid_utils import (
    generate_perimeter_points, sanitize_filename, save_points,
    load_edit_profile
)
from settings.config.environment_config import environment
from PyQt5.QtGui import QGuiApplication

SETTINGS_DIR = "settings"
POINT_RADIUS = 6
GRID_DIV = 10

def get_point_path(display_name):
    return os.path.join(SETTINGS_DIR, f"{sanitize_filename(display_name)}_warp_map_points.json")

def get_screen_index(display_name):
    screens = QGuiApplication.screens()
    edit_display = load_edit_profile()
    active_screens = [s for s in screens if s.name() != edit_display]
    for idx, screen in enumerate(active_screens):
        if screen.name() == display_name:
            return idx
    return None

class EditorCanvas(tk.Canvas):
    def __init__(self, master, display_name, width, height):
        super().__init__(master, width=width, height=height, bg="black")
        self.display_name = display_name
        self.w, self.h = width, height
        self.points = self.load_initial_points()
        self.dragging_point = None

        self.bind("<ButtonPress-1>", self.on_press)
        self.bind("<B1-Motion>", self.on_drag)
        self.bind("<ButtonRelease-1>", self.on_release)

        self.draw()

    def draw(self):
        self.delete("all")
        for i in range(len(self.points)):
            x, y = self.points[i]
            self.create_oval(x - POINT_RADIUS, y - POINT_RADIUS, x + POINT_RADIUS, y + POINT_RADIUS, fill="red")
            x2, y2 = self.points[(i + 1) % len(self.points)]
            self.create_line(x, y, x2, y2, fill="green", width=2)

    def on_press(self, event):
        for i, (x, y) in enumerate(self.points):
            if abs(event.x - x) < POINT_RADIUS and abs(event.y - y) < POINT_RADIUS:
                self.dragging_point = i
                return

    def on_drag(self, event):
        if self.dragging_point is not None:
            self.points[self.dragging_point] = [event.x, event.y]
            self.draw()

    def on_release(self, event):
        self.dragging_point = None

    def load_initial_points(self):
        path = get_point_path(self.display_name)
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return generate_perimeter_points(self.w, self.h, GRID_DIV)

    def save(self):
        save_points(self.display_name, self.points, mode="warp_map")
        print(f"✅ 保存: {get_point_path(self.display_name)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode")
    parser.add_argument("--display", required=True)
    parser.add_argument("--x", type=int)
    parser.add_argument("--y", type=int)
    parser.add_argument("--w", type=int)
    parser.add_argument("--h", type=int)
    args = parser.parse_args()

    root = tk.Tk()
    root.title(f"{args.display} - 自由変形モード")
    root.geometry(f"{args.w}x{args.h}+{args.x}+{args.y}")

    main_frame = tk.Frame(root)
    main_frame.pack(fill="both", expand=True)

    canvas = EditorCanvas(main_frame, args.display, args.w, args.h)
    canvas.pack(fill="both", expand=True)

    root.mainloop()

if __name__ == "__main__":
    main()
