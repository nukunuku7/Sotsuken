# grid_editor_perspective.py

import argparse
import tkinter as tk
import json
import sys
import os
import threading
import time

def ensure_module_path():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    return base_dir

BASE_DIR = ensure_module_path()
TEMP_DIR = os.path.join(BASE_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

from editor.grid_utils import (
    generate_perspective_points, save_points,
    get_point_path, sanitize_filename
)

POINT_RADIUS = 8

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
        self.create_polygon(*sum(self.points, []), outline="green", fill="", width=2)
        for x, y in self.points:
            self.create_oval(x - POINT_RADIUS, y - POINT_RADIUS,
                             x + POINT_RADIUS, y + POINT_RADIUS, fill="red")

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
        path = get_point_path(self.display_name, mode="perspective")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return generate_perspective_points(self.display_name)

    def save(self):
        save_points(self.display_name, self.points, mode="perspective")
        print(f"✅ 保存: {get_point_path(self.display_name, mode='perspective')}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--display", required=True)
    parser.add_argument("--x", type=int)
    parser.add_argument("--y", type=int)
    parser.add_argument("--w", type=int)
    parser.add_argument("--h", type=int)
    args = parser.parse_args()

    root = tk.Tk()
    root.geometry(f"{args.w}x{args.h}+{args.x}+{args.y}")
    root.overrideredirect(True)
    root.bind("<Escape>", lambda e: root.destroy())

    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True)
    canvas = EditorCanvas(frame, args.display, args.w, args.h)
    canvas.pack(fill="both", expand=True)

    lock_path = os.path.join(TEMP_DIR, f"editor_active_{sanitize_filename(args.display, 'perspective')}.lock")
    def watch_lock():
        while True:
            time.sleep(0.5)
            if not os.path.exists(lock_path):
                canvas.save()
                root.destroy()
                break

    threading.Thread(target=watch_lock, daemon=True).start()
    root.mainloop()

if __name__ == "__main__":
    main()
