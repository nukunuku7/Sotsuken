# media_player_multi.py
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
    def __init__(self, source_screen, target_screen,
                 offset_x, virtual_size, warp_info):
        super().__init__()

        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_DeleteOnClose)

        g = target_screen.geometry()
        self.setGeometry(g.x(), g.y(), g.width(), g.height())

        self.offset_x = offset_x
        self.virtual_size = virtual_size
        self.warp_info = warp_info

        self.target_width = g.width()
        self.target_height = g.height()

        self.sct = mss.mss()
        sg = source_screen.geometry()

        # ★ 短冊単位でキャプチャ
        self.monitor = {
            "top": sg.y(),
            "left": sg.x(),
            "width": self.virtual_size[0],
            "height": self.virtual_size[1],
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

                // 範囲外ガード（重要）
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

        # --- 映像テクスチャ（短冊サイズ）
        self.video_tex = self.ctx.texture(
            self.virtual_size, 4
        )
        self.video_tex.swizzle = "BGRA"
        self.video_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # --- warp map（すでに短冊化済み）
        map_x, map_y = self.warp_info
        uv = np.dstack([
            map_x / float(self.virtual_size[0]),
            map_y / float(self.virtual_size[1])
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
    parser.add_argument("--blend", action="store_true")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    screens = {get_virtual_id(s.name()): s for s in QGuiApplication.screens()}
    source = screens[get_virtual_id(args.source)]

    total_w = sum(screens[get_virtual_id(t)].geometry().width() for t in args.targets)
    max_h = max(screens[get_virtual_id(t)].geometry().height() for t in args.targets)

    offset = 0
    windows = []

    for t in args.targets:
        scr = screens[get_virtual_id(t)]
        w = scr.geometry().width()
        h = scr.geometry().height()

        map_x, map_y = prepare_warp(
            t,
            args.mode,
            src_size=(source.geometry().width(), source.geometry().height()),
            load_points_func=load_points,
            log_func=log
        )

        win = GLDisplayWindow(
            source, scr, offset, (total_w, max_h), (map_x, map_y)
        )
        win.show()
        windows.append(win)

        offset += w

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
