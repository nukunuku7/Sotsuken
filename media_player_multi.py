# media_player_multi.py (FINAL)

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

        # === grid points (normalized UV) ===
        pts = self.perspective_pts if mode == "perspective" else self.warp_pts
        pts[:, 0] /= self.full_w
        pts[:, 1] /= self.full_h
        self.grid_uv = pts

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(16)  # ~60fps

    # -----------------------------------------------------
    def initializeGL(self):
        self.ctx = moderngl.create_context()

        if self.enable_blend:
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (
                moderngl.SRC_ALPHA,
                moderngl.ONE_MINUS_SRC_ALPHA,
            )

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

                uniform int mode;
                uniform vec4 crop_rect;
                uniform float left_fade;
                uniform float right_fade;
                uniform int enable_blend;

                void main() {
                    vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);

                    if (mode == 1) {
                        uv = texture(warp_tex, v_uv).rg;
                        if (uv.x < 0.0 || uv.x > 1.0 ||
                            uv.y < 0.0 || uv.y > 1.0)
                            discard;
                    }

                    if (uv.x < crop_rect.x || uv.x > crop_rect.z ||
                        uv.y < crop_rect.y || uv.y > crop_rect.w)
                        discard;

                    vec4 color = texture(video_tex, uv);

                    if (enable_blend == 1) {
                        float a = 1.0;
                        if (v_uv.x < left_fade)
                            a = v_uv.x / left_fade;
                        else if (v_uv.x > right_fade)
                            a = (1.0 - v_uv.x) / (1.0 - right_fade);
                        color.a *= clamp(a, 0.0, 1.0);
                    }

                    fragColor = color;
                }
            """
        )

        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog, [(self.vbo, "2f 2f", "in_pos", "in_uv")]
        )

        self.video_tex = self.ctx.texture(
            (self.full_w, self.full_h), 4
        )
        self.video_tex.swizzle = "BGRA"

        uv = np.dstack([
            self.warp_npz["map_x"],
            self.warp_npz["map_y"]
        ]).astype("f4")

        self.warp_tex = self.ctx.texture(
            (self.full_w, self.full_h), 2, uv.tobytes(), dtype="f4"
        )

        u0, v0 = self.grid_uv.min(axis=0)
        u1, v1 = self.grid_uv.max(axis=0)
        self.prog["crop_rect"].value = (u0, v0, u1, v1)

        overlap = self.slice["overlap"]
        w = self.slice["w"]

        self.prog["left_fade"].value = overlap / w
        self.prog["right_fade"].value = 1.0 - overlap / w
        self.prog["enable_blend"].value = 1 if self.enable_blend else 0
        self.prog["mode"].value = 0 if self.mode == "perspective" else 1

    # -----------------------------------------------------
    def paintGL(self):
        img = self.sct.grab(self.monitor)
        frame = np.array(img, dtype=np.uint8)

        frame = cv2.resize(
            frame, (self.full_w, self.full_h),
            interpolation=cv2.INTER_LINEAR
        )

        self.video_tex.write(frame.tobytes())

        cols = self.slice["count"]
        idx = self.slice["index"]
        gw = self.width() // cols
        self.ctx.viewport = (idx * gw, 0, gw, self.height())

        self.video_tex.use(0)
        self.warp_tex.use(1)
        self.vao.render(moderngl.TRIANGLE_STRIP)
