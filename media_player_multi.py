import sys
import json
import argparse
import mss
import moderngl
import numpy as np
from pathlib import Path

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtWidgets import QApplication, QOpenGLWidget

from editor.grid_utils import get_virtual_id


# =========================================================
# Profile Loader
# =========================================================
def load_profiles(virt_id, simulator_id, full_size):
    base = Path("config")

    # --- warp map (simulator 기준) ---
    warp_npz = np.load(
        base / "warp_cache" /
        f"ScreenSimulatorSet_{simulator_id}_map_{full_size[0]}x{full_size[1]}.npz"
    )

    # --- perspective ---
    with open(base / "projector_profiles" /
              f"{virt_id}_perspective_points.json", encoding="utf-8") as f:
        perspective = json.load(f)

    # --- warp grid ---
    with open(base / "projector_profiles" /
              f"{virt_id}_warp_map_points.json", encoding="utf-8") as f:
        warp_points = json.load(f)

    return warp_npz, perspective, warp_points


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
        self.enable_blend = slice_geom["count"] > 1

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

        if self.enable_blend:
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (
                moderngl.SRC_ALPHA,
                moderngl.ONE_MINUS_SRC_ALPHA,
            )

        # ===== Fullscreen quad =====
        vertices = np.array([
            -1, -1, 0, 0,
             1, -1, 1, 0,
            -1,  1, 0, 1,
             1,  1, 1, 1,
        ], dtype="f4")

        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                layout (location=0) in vec2 in_pos;
                layout (location=1) in vec2 in_uv;
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

                uniform int mode; // 0=perspective, 1=warp
                uniform float left_fade;
                uniform float right_fade;
                uniform int enable_blend;

                void main() {
                    vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);

                    if (mode == 1) {
                        uv = texture(warp_tex, v_uv).rg;
                        if (uv.x < 0 || uv.x > 1 ||
                            uv.y < 0 || uv.y > 1)
                            discard;
                    }

                    vec4 color = texture(video_tex, uv);

                    if (enable_blend == 1) {
                        float a = 1.0;
                        if (v_uv.x < left_fade)
                            a = v_uv.x / left_fade;
                        else if (v_uv.x > right_fade)
                            a = (1.0 - v_uv.x) / (1.0 - right_fade);
                        color.a *= clamp(a,0,1);
                    }

                    fragColor = color;
                }
            """
        )

        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog, [(self.vbo, "2f 2f", 0, 1)]
        )

        # ===== video texture (slice size) =====
        self.video_tex = self.ctx.texture(
            (self.slice["w"], self.slice["h"]), 4
        )
        self.video_tex.swizzle = "BGRA"

        # ===== warp texture (full resolution) =====
        map_x = self.warp_npz["map_x"]
        map_y = self.warp_npz["map_y"]

        uv = np.dstack([
            map_x / self.full_w,
            map_y / self.full_h
        ]).astype("f4")

        self.warp_tex = self.ctx.texture(
            (self.full_w, self.full_h), 2, uv.tobytes(), dtype="f4"
        )

        # ===== uniforms =====
        overlap = self.slice["overlap"]
        w = self.slice["w"]

        self.prog["left_fade"].value = overlap / w
        self.prog["right_fade"].value = 1.0 - overlap / w
        self.prog["enable_blend"].value = 1 if self.enable_blend else 0
        self.prog["mode"].value = 0 if self.mode == "perspective" else 1

    # -----------------------------------------------------
    def paintGL(self):
        img = self.sct.grab(self.monitor)
        self.video_tex.write(img.raw)

        # ---- grid viewport ----
        cols = self.slice["count"]
        idx = self.slice["index"]
        gw = self.width() // cols
        self.ctx.viewport = (idx * gw, 0, gw, self.height())

        self.video_tex.use(0)
        self.warp_tex.use(1)
        self.vao.render(moderngl.TRIANGLE_STRIP)


# =========================================================
# main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--mode", required=True,
                        choices=["perspective", "map"])
    args = parser.parse_args()

    app = QApplication(sys.argv)
    screens = QGuiApplication.screens()

    src = next(s for s in screens if s.name() == args.source)
    geom = src.geometry()

    slice_w = geom.width() // len(args.targets)
    windows = []

    # 左→右順で simulator_id を割り当て
    for sim_idx, virt in enumerate(args.targets):
        tgt = next(s for s in screens if get_virtual_id(s.name()) == virt)

        profiles = load_profiles(
            virt_id=virt,
            simulator_id=sim_idx + 1,
            full_size=(geom.width(), geom.height())
        )

        sg = {
            "x": geom.x() + sim_idx * slice_w,
            "y": geom.y(),
            "w": slice_w,
            "h": geom.height(),
            "index": sim_idx,
            "count": len(args.targets),
            "overlap": 100,
        }

        win = GLDisplayWindow(
            src, tgt, sg, args.mode, profiles
        )
        win.show()
        windows.append(win)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
