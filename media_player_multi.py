import sys
import mss
import cv2
import signal
import argparse
import moderngl
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtWidgets import QOpenGLWidget, QApplication

from editor.grid_utils import load_points, log, get_virtual_id
from warp_engine import (
    prepare_warp,
    convert_maps_to_uv_texture_data,
)

from pathlib import Path

def get_simulator_name_for_screen(screen, edit_display_name):
    if screen.name() == edit_display_name:
        raise RuntimeError("Edit display must not be used as warp target")

    screens = [
        s for s in QGuiApplication.screens()
        if s.name() != edit_display_name
    ]
    screens = sorted(screens, key=lambda s: s.geometry().x())
    idx = screens.index(screen) + 1
    return f"ScreenSimulatorSet_{idx}"

def load_warp_uv_texture(ctx, uv_path, w, h):
    uv = np.load(uv_path).astype("f4")  # (H, W, 2)
    assert uv.shape == (h, w, 2), f"UV shape mismatch: {uv.shape}"
    return ctx.texture((w, h), 2, uv.tobytes(), dtype="f4")

def create_identity_uv(ctx, w, h):
    xs = np.linspace(0, 1, w, dtype=np.float32)
    ys = np.linspace(0, 1, h, dtype=np.float32)
    u, v = np.meshgrid(xs, ys)
    uv = np.dstack([u, v]).astype("f4")
    return ctx.texture((w, h), 2, uv.tobytes(), dtype="f4")

class GLDisplayWindow(QOpenGLWidget):
    def __init__(self, source_screen, target_screen, mode,
                 warp_info_all=None,
                 source_geometry=None,
                 edit_display_name=None):
        super().__init__()

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_DeleteOnClose)

        g = target_screen.geometry()
        self.setFixedSize(g.width(), g.height())
        self.move(g.x(), g.y())

        self.source_geometry = source_geometry
        self.warp_info_all = warp_info_all

        self.slice_index = source_geometry.get("index", 0)
        self.slice_count = source_geometry.get("count", 1)
        self.overlap_px = source_geometry.get("overlap", 0)

        self.enable_blend = self.slice_count > 1

        self.sct = mss.mss()
        self.monitor = {
            "top": source_geometry["y"],
            "left": source_geometry["x"],
            "width": source_geometry["w"],
            "height": source_geometry["h"],
        }

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(16)

        self.edit_display_name = edit_display_name
        self.target_screen = target_screen

    def initializeGL(self):
        print("[INIT] OpenGL initialization start")
        print("[INIT] Warp map loading start")
        print("[INIT] Ready")

        self.ctx = moderngl.create_context()

        # ===== フルスクリーンクアッド =====
        vertices = np.array([
            -1.0, -1.0, 0.0, 1.0,
             1.0, -1.0, 1.0, 1.0,
            -1.0,  1.0, 0.0, 0.0,
             1.0,  1.0, 1.0, 0.0,
        ], dtype='f4')

        if self.enable_blend:
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (
                moderngl.SRC_ALPHA,
                moderngl.ONE_MINUS_SRC_ALPHA,
            )

        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_vert;
                in vec2 in_text;
                out vec2 v_uv;
                void main() {
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                    v_uv = in_text;
                }
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D original_tex;
                uniform sampler2D warp_uv_tex;

                uniform float slice_left;
                uniform float slice_right;
                uniform int enable_blend;

                in vec2 v_uv;
                out vec4 fragColor;

                void main() {
                    vec2 warped_uv = texture(warp_uv_tex, v_uv).rg;

                    // 範囲外は完全透明
                    if (warped_uv.x < 0.0 || warped_uv.x > 1.0 ||
                        warped_uv.y < 0.0 || warped_uv.y > 1.0) {
                        fragColor = vec4(0.0);
                        return;
                    }

                    vec4 color = texture(original_tex, warped_uv);

                    if (enable_blend == 1) {
                        float alpha = 1.0;
                        if (warped_uv.x < slice_left) {
                            alpha = smoothstep(0.0, slice_left, warped_uv.x);
                        } else if (warped_uv.x > slice_right) {
                            alpha = smoothstep(1.0, slice_right, warped_uv.x);
                        }
                        color.a *= alpha;
                    }

                    fragColor = color;
                }
            """
        )

        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, '2f 2f', 'in_vert', 'in_text')]
        )

        # ===== キャプチャテクスチャ =====
        w = self.monitor["width"]
        h = self.monitor["height"]
        self.texture_video = self.ctx.texture((w, h), 4)
        self.texture_video.swizzle = "BGRA"

        # ===== Warp UV テクスチャ =====
        simulator_name = get_simulator_name_for_screen(
            self.target_screen,
            self.edit_display_name
        )

        map_pair = prepare_warp(
            display_name=simulator_name,
            mode="map",
            src_size=(w, h),
            log_func=log,
        )
        print(
            f"[WARP] using {simulator_name}_map_{w}x{h}.npz"
        )

        if map_pair is None:
            raise RuntimeError(
                "Warp map missing. Please run precompute_warp_maps.py first."
            )

        map_x, map_y = map_pair
        uv_bytes = convert_maps_to_uv_texture_data(map_x, map_y, w, h)
        self.texture_warp = self.ctx.texture(
            (w, h), 2, uv_bytes, dtype="f4"
        )

        print(f"[INFO] warp map applied: {simulator_name}")

        self.prog["original_tex"].value = 0
        self.prog["warp_uv_tex"].value = 1

        slice_w = 1.0 / self.slice_count
        overlap = self.overlap_px / w

        left = self.slice_index * slice_w
        right = (self.slice_index + 1) * slice_w
        self.prog["slice_left"].value = left + overlap
        self.prog["slice_right"].value = right - overlap
        self.prog["enable_blend"].value = 1 if self.enable_blend else 0

    def paintGL(self):
        dpr = self.devicePixelRatioF()
        self.ctx.viewport = (
            0, 0,
            int(self.width() * dpr),
            int(self.height() * dpr),
        )

        img = self.sct.grab(self.monitor)
        self.texture_video.write(img.raw)

        self.texture_video.use(0)
        self.texture_warp.use(1)
        self.vao.render(moderngl.TRIANGLE_STRIP)

# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    print("[BOOT] media_player_multi starting")

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--blend", action="store_true")
    args = parser.parse_args()

    print("[BOOT] args:", args)

    app = QApplication(sys.argv)

    screens = QGuiApplication.screens()
    print("[BOOT] detected screens:")
    for s in screens:
        g = s.geometry()
        print(f"  - {s.name()} {g.width()}x{g.height()} ({g.x()},{g.y()})")

    # source screen
    source_screen = next(
        s for s in screens if s.name() == args.source
    )

    windows = []

    for idx, target_virt in enumerate(args.targets):
        target_screen = next(
            s for s in screens if get_virtual_id(s.name()) == target_virt
        )

        geom = source_screen.geometry()
        source_geometry = {
            "x": geom.x(),
            "y": geom.y(),
            "w": geom.width(),
            "h": geom.height(),
            "index": idx,
            "count": len(args.targets),
            "overlap": 0,
        }

        print(
            f"[BOOT] creating window for {target_screen.name()} "
            f"(slice {idx+1}/{len(args.targets)})"
        )

        win = GLDisplayWindow(
            source_screen=source_screen,
            target_screen=target_screen,
            mode=args.mode,
            source_geometry=source_geometry,
            edit_display_name=args.source,
        )
        win.show()
        windows.append(win)

    print("[BOOT] entering Qt event loop")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
