import argparse
import tkinter as tk
import sys
import os
import threading
import time
from pathlib import Path

def ensure_module_path():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    return base_dir

BASE_DIR = ensure_module_path()
TEMP_DIR = Path(BASE_DIR) / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

from editor.grid_utils import (
    save_points, load_points, generate_border_points,
    get_virtual_id, get_point_path,
    claim_next_point, mark_point_done
)

POINT_RADIUS = 6


class EditorCanvas(tk.Canvas):
    def __init__(self, master, display_name, width, height):
        super().__init__(master, width=width, height=height, bg="black")

        self.display_name = display_name
        self.virt = get_virtual_id(display_name)
        self.w, self.h = width, height

        self.saved_once = False
        self.points = self.load_initial_points()

        # --- 最初の点を自動でマウスに追従 ---
        self.dragging_point = 0  # 最初の点
        self.follow_mouse_loop()

        self.bind("<ButtonPress-1>", self.on_press)
        self.bind("<B1-Motion>", self.on_drag)
        self.bind("<ButtonRelease-1>", self.on_release)

        self.draw()

    def follow_mouse_loop(self):
        if self.dragging_point is not None:
            x = self.winfo_pointerx() - self.winfo_rootx()
            y = self.winfo_pointery() - self.winfo_rooty()
            nx = max(0, min(self.w, x))
            ny = max(0, min(self.h, y))
            self.points[self.dragging_point] = [nx, ny]
            self.draw()
        self.after(10, self.follow_mouse_loop)  # 10ms ごとに更新

    def draw(self):
        self.delete("all")
        if not self.points:
            return

        for i in range(len(self.points)):
            x, y = self.points[i]

            self.create_oval(
                x - POINT_RADIUS, y - POINT_RADIUS,
                x + POINT_RADIUS, y + POINT_RADIUS,
                fill="red"
            )

            x2, y2 = self.points[(i + 1) % len(self.points)]
            self.create_line(x, y, x2, y2, fill="green", width=2)

    def on_press(self, event):
        for i, (x, y) in enumerate(self.points):
            if abs(event.x - x) < POINT_RADIUS and abs(event.y - y) < POINT_RADIUS:
                self.dragging_point = i
                return

    def on_drag(self, event):
        if self.dragging_point is not None:
            nx = max(0, min(self.w, event.x))
            ny = max(0, min(self.h, event.y))
            self.points[self.dragging_point] = [nx, ny]
            self.draw()

    def on_release(self, event):
        if self.dragging_point is not None:
            # 現在の点をセッションに記録
            mark_point_done(self.virt, self.dragging_point)
            
            # 次の点を取得
            next_index = claim_next_point(self.virt)
            if next_index is not None:
                self.dragging_point = next_index  # 次の点をマウスに追従
            else:
                self.dragging_point = None  # すべて完了

    def load_initial_points(self):
        existing = load_points(self.virt, mode="warp_map")
        if existing:
            return existing  # ← 36点が入っていればそのまま

        # 外周36点を生成
        pts = generate_border_points(self.w, self.h, divisions=10)
        save_points(self.virt, pts, mode="warp_map")
        return pts

    def save(self):
        if self.saved_once:
            return
        self.saved_once = True

        save_points(self.virt, self.points, mode="warp_map")
        print(f"保存: {get_point_path(self.virt, 'warp_map')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--display", required=True)
    parser.add_argument("--x", type=int, default=0)
    parser.add_argument("--y", type=int, default=0)
    parser.add_argument("--w", type=int, default=1920)
    parser.add_argument("--h", type=int, default=1080)
    args = parser.parse_args()

    root = tk.Tk()
    root.geometry(f"{args.w}x{args.h}+{args.x}+{args.y}")
    root.overrideredirect(True)
    root.bind("<Escape>", lambda e: root.destroy())

    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True)

    canvas = EditorCanvas(frame, args.display, args.w, args.h)
    canvas.pack(fill="both", expand=True)

    virt = get_virtual_id(args.display)
    lock_path = TEMP_DIR / f"editor_active_{virt}_warp_map.lock"

    with open(lock_path, "w", encoding="utf-8") as f:
        f.write("active")

    def watch_lock():
        while True:
            time.sleep(0.3)
            if not lock_path.exists():
                try:
                    canvas.save()
                except Exception as e:
                    print(f"[ERROR] save failed: {e}")
                try:
                    root.destroy()
                except:
                    pass
                break

    threading.Thread(target=watch_lock, daemon=True).start()
    root.mainloop()


if __name__ == "__main__":
    main()
