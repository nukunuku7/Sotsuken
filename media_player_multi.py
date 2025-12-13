import sys
import argparse
import numpy as np
import mss
import moderngl
import signal

from PyQt5.QtWidgets import QApplication, QOpenGLWidget
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtCore import QTimer, Qt

from editor.grid_utils import load_points, log, get_virtual_id
from warp_engine import prepare_warp


class GLDisplayWindow(QOpenGLWidget):
    def __init__(self, source_screen, target_screen, warp_info):
        super().__init__()

        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_DeleteOnClose)

        g = target_screen.geometry()
        self.setGeometry(g.x(), g.y(), g.width(), g.height())

        self.target_width = g.width()
        self.target_height = g.height()

        self.source_screen = source_screen
        self.warp_info = warp_info

        self.sct = mss.mss()
        sg = source_screen.geometry()

        # ★ 編集用ディスプレイ全体を常にキャプチャ
        self.monitor = {
            "top": sg.y(),
            "left": sg.x(),
            "width": sg.width(),
            "height": sg.height(),
        }

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(8)  # ~120fps

    def initializeGL(self):
        self.ctx = moderngl.create_context()

        vertices = np.array([
            -1, -1, 0, 1,
             1, -1, 1, 1,
            -1,  1, 0, 0,
             1,  1, 1, 0,
        ], dtype="f4")

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
            uniform sampler2D video_tex;
            uniform sampler2D warp_tex;

            in vec2 v_uv;
            out vec4 frag_color;

            void main() {
                vec2 src_uv = texture(warp_tex, v_uv).rg;

                if (src_uv.x < 0.0 || src_uv.x > 1.0 ||
                    src_uv.y < 0.0 || src_uv.y > 1.0) {
                    frag_color = vec4(0.0, 0.0, 0.0, 1.0);
                } else {
                    frag_color = texture(video_tex, src_uv);
                }
            }
            """
        )

        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog, [(self.vbo, "2f 2f", "in_vert", "in_text")]
        )

        # --- 映像テクスチャ（編集用ディスプレイ全体）
        sg = self.source_screen.geometry()
        self.video_tex = self.ctx.texture((sg.width(), sg.height()), 4)
        self.video_tex.swizzle = "BGRA"
        self.video_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # --- warp map（短冊）
        map_x, map_y = self.warp_info
        uv = np.dstack([
            map_x / float(sg.width()),
            map_y / float(sg.height())
        ]).astype("f4")

        h, w = map_x.shape
        self.warp_tex = self.ctx.texture((w, h), 2, uv.tobytes(), dtype="f4")
        self.warp_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)

        self.prog["video_tex"].value = 0
        self.prog["warp_tex"].value = 1

    def paintGL(self):
        img = self.sct.grab(self.monitor)
        self.video_tex.write(img.raw)
        self.video_tex.use(0)
        self.warp_tex.use(1)
        self.vao.render(moderngl.TRIANGLE_STRIP)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--mode", default="warp_map")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    screens = {get_virtual_id(s.name()): s for s in QGuiApplication.screens()}
    source = screens[get_virtual_id(args.source)]

    num_targets = len(args.targets)
    source_w = source.geometry().width()
    source_h = source.geometry().height()

    slice_w = source_w // num_targets

    windows = []

    for i, t in enumerate(args.targets):
        scr = screens[get_virtual_id(t)]
        w = scr.geometry().width()
        h = scr.geometry().height()

        full_map_x, full_map_y = prepare_warp(
            t,
            args.mode,
            src_size=(source_w, source_h),
            load_points_func=load_points,
            log_func=log
        )

        x0 = i * slice_w
        x1 = x0 + slice_w

        map_x = full_map_x[:h, x0:x1]
        map_y = full_map_y[:h, x0:x1]

        win = GLDisplayWindow(
            source,
            scr,
            (map_x, map_y)
        )
        win.show()
        windows.append(win)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
