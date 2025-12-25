# media_player_multi.py (FIXED FINAL)

import sys
import json
import argparse
import mss
import moderngl
import numpy as np
import cv2
from pathlib import Path

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtWidgets import QApplication, QOpenGLWidget

from editor.grid_utils import get_virtual_id


# =========================================================
# Profile Loader
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"

def load_profiles(virt_id, simulator_index, full_size):
    w, h = full_size

    warp_npz = np.load(
        CONFIG_DIR / "warp_cache" /
        f"ScreenSimulatorSet_{simulator_index}_map_{w}x{h}.npz"
    )

    with open(
        CONFIG_DIR / "projector_profiles" /
        f"{virt_id}_perspective_points.json",
        encoding="utf-8"
    ) as f:
        perspective_pts = np.array(json.load(f), np.float32)

    with open(
        CONFIG_DIR / "projector_profiles" /
        f"{virt_id}_warp_map_points.json",
        encoding="utf-8"
    ) as f:
        warp_pts = np.array(json.load(f), np.float32)

    return warp_npz, perspective_pts, warp_pts


# =========================================================
# OpenGL Window
# =========================================================
class GLDisplayWindow(QOpenGLWidget):
    def __init__(self, source_screen, target_screen,
                 slice_geom, mode, profiles):
        super().__init__()

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_DeleteOnClose)

        tg = target_screen.geometry()
        self.setFixedSize(tg.width(), tg.height())
        self.move(tg.x(), tg.y())

        self.slice = slice_geom
        self.mode = mode

        self.full_w = source_screen.geometry().width()
        self.full_h = source_screen.geometry().height()

        self.sct = mss.mss()
        self.monitor = {
            "top": slice_geom["y"],
            "left": slice_geom["x"],
            "width": slice_geom["w"],
            "height": slice_geom["h"],
        }

        self.warp_npz, self.perspective_pts, self.warp_pts = profiles

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(16)

    # -----------------------------------------------------
    def initializeGL(self):
        self.ctx = moderngl.create_context()

        vertices = np.array([
            -1, -1, 0, 0,
             1, -1, 1, 0,
            -1,  1, 0, 1,
             1,  1, 1, 1,
        ], dtype="f4")

        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                in vec2 in_uv;
                out vec2 v_uv;
                void main() {
                    gl_Position = vec4(in_pos,0,1);
                    v_uv = in_uv;
                }
            """,
            fragment_shader="""
                #version 330
                in vec2 v_uv;
                out vec4 fragColor;

                uniform sampler2D video_tex;
                uniform sampler2D warp_tex;
                uniform vec4 crop_rect;
                uniform int mode;

                void main() {

                    vec2 uv;

                    if (mode == 1) {
                        // warp_map: map_x/map_y は video_tex 用UV
                        uv = texture(warp_tex, v_uv).rg;
                    } else {
                        // perspective: そのまま
                        uv = vec2(v_uv.x, 1.0 - v_uv.y);
                    }

                    // ---- crop (共通)
                    if (uv.x < crop_rect.x || uv.x > crop_rect.z ||
                        uv.y < crop_rect.y || uv.y > crop_rect.w)
                        discard;

                    // ---- safety
                    if (uv.x < 0.0 || uv.x > 1.0 ||
                        uv.y < 0.0 || uv.y > 1.0)
                        discard;

                    fragColor = vec4(uv, 0.0, 1.0);
                }

            """
        )

        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog, [(self.vbo, "2f 2f", "in_pos", "in_uv")]
        )

        # ---- video texture
        self.video_tex = self.ctx.texture(
            (self.full_w, self.full_h), 4
        )
        self.video_tex.swizzle = "BGRA"

        # ---- warp texture (map mode only)
        map_x = self.warp_npz["map_x"] / (self.full_w - 1)
        map_y = self.warp_npz["map_y"] / (self.full_h - 1)

        uv = np.dstack([map_x, map_y]).astype("f4")

        self.warp_tex = self.ctx.texture(
            (self.full_w, self.full_h), 2, uv.tobytes(), dtype="f4"
        )

        # ---- grid → crop_rect
        pts = self.perspective_pts if self.mode == "perspective" else self.warp_pts
        pts = pts.copy()
        pts[:, 0] /= self.full_w
        pts[:, 1] /= self.full_h

        sx0 = self.slice["x"] / self.full_w
        sx1 = (self.slice["x"] + self.slice["w"]) / self.full_w

        mask = (pts[:, 0] >= sx0) & (pts[:, 0] <= sx1)
        pts = pts[mask]

        if len(pts) >= 2:
            u0, v0 = pts.min(axis=0)
            u1, v1 = pts.max(axis=0)
        else:
            # フォールバック（絶対に真っ黒にしない）
            u0, v0, u1, v1 = sx0, 0.0, sx1, 1.0

        self.prog["crop_rect"].value = (u0, v0, u1, v1)
        self.prog["mode"].value = 0 if self.mode == "perspective" else 1

    # -----------------------------------------------------
    def paintGL(self):
        img = self.sct.grab(self.monitor)
        frame = np.array(img, dtype=np.uint8)
        frame = cv2.resize(frame, (self.full_w, self.full_h))

        self.video_tex.write(frame.tobytes())
        self.video_tex.use(0)
        if self.mode == "map":
            self.warp_tex.use(1)

        self.vao.render(moderngl.TRIANGLE_STRIP)


# =========================================================
# Main Entry Point
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--mode", required=True, choices=["perspective", "map"])
    parser.add_argument("--targets", nargs="+", required=True)
    args = parser.parse_args()

    app = QApplication(sys.argv)

    screens = QGuiApplication.screens()
    screen_map = {s.name(): s for s in screens}

    if args.source not in screen_map:
        print("[ERROR] Source screen not found")
        sys.exit(1)

    source_screen = screen_map[args.source]
    full_w = source_screen.geometry().width()
    full_h = source_screen.geometry().height()

    windows = []

    for idx, tgt in enumerate(args.targets):
        if tgt not in screen_map:
            print(f"[WARN] Target screen not found: {tgt}")
            continue

        virt_id = get_virtual_id(tgt)

        profiles = load_profiles(
            virt_id=virt_id,
            simulator_index=idx + 1,
            full_size=(full_w, full_h),
        )

        slice_geom = {
            "index": idx,
            "count": len(args.targets),
            "x": int(idx * full_w / len(args.targets)),
            "y": 0,
            "w": int(full_w / len(args.targets)),
            "h": full_h,
        }

        win = GLDisplayWindow(
            source_screen,
            screen_map[tgt],
            slice_geom,
            args.mode,
            profiles,
        )
        win.show()
        windows.append(win)

    if not windows:
        print("[ERROR] No windows created")
        sys.exit(1)

    print(f"[OK] Launched {len(windows)} windows")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
