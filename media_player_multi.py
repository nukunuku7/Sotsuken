# media_player_multi.py
import sys
import argparse
import mss
import moderngl
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtWidgets import QOpenGLWidget, QApplication

from editor.grid_utils import get_virtual_id
from warp_engine import prepare_warp, convert_maps_to_uv_texture_data


# ------------------------------------------------------------
# GL Window (1 window = 1 slice)
# ------------------------------------------------------------
class GLDisplayWindow(QOpenGLWidget):
    def __init__(
        self,
        source_geom,
        target_screen,
        display_id,
        mode,
        slice_index,
        slice_count,
        overlap_px,
    ):
        super().__init__()

        g = target_screen.geometry()
        self.setGeometry(g)
        self.setWindowFlags(Qt.FramelessWindowHint)

        self.display_id = display_id
        self.mode = mode
        self.slice_index = slice_index
        self.slice_count = slice_count
        self.overlap_px = overlap_px

        self.src_geom = source_geom
        self.src_w = source_geom["width"]
        self.src_h = source_geom["height"]

        self.sct = mss.mss()

    # --------------------------------------------------------
    def initializeGL(self):
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (
            moderngl.SRC_ALPHA,
            moderngl.ONE_MINUS_SRC_ALPHA,
        )

        # fullscreen quad
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
                    gl_Position = vec4(in_pos, 0.0, 1.0);
                    v_uv = in_uv;
                }
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D video_tex;
                uniform sampler2D warp_tex;

                uniform float slice_l;
                uniform float slice_r;

                in vec2 v_uv;
                out vec4 fragColor;

                void main() {
                    vec2 uv = texture(warp_tex, v_uv).rg;

                    if (uv.x < 0.0 || uv.x > 1.0 ||
                        uv.y < 0.0 || uv.y > 1.0) {
                        fragColor = vec4(0.0);
                        return;
                    }

                    vec4 col = texture(video_tex, uv);

                    float x = uv.x;
                    float a = 1.0;
                    if (x < slice_l)
                        a = smoothstep(0.0, slice_l, x);
                    else if (x > slice_r)
                        a = smoothstep(1.0, slice_r, x);

                    fragColor = vec4(col.rgb, col.a * a);
                }
            """
        )

        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, "2f 2f", "in_pos", "in_uv")]
        )

        # video texture
        self.video_tex = self.ctx.texture(
            (self.src_w, self.src_h), 4
        )
        self.video_tex.swizzle = "BGRA"

        # warp map (1回だけ)
        map_x, map_y = prepare_warp(
            display_name=self.display_id,
            mode=self.mode,
            src_size=(self.src_w, self.src_h),
        )

        uv_bytes = convert_maps_to_uv_texture_data(
            map_x, map_y, self.src_w, self.src_h
        )

        self.warp_tex = self.ctx.texture(
            (self.src_w, self.src_h),
            2,
            uv_bytes,
            dtype="f4",
        )

        self.video_tex.use(0)
        self.warp_tex.use(1)

        self.prog["video_tex"].value = 0
        self.prog["warp_tex"].value = 1

        # slice parameters
        slice_w = 1.0 / self.slice_count
        overlap = self.overlap_px / self.src_w

        l = self.slice_index * slice_w + overlap
        r = (self.slice_index + 1) * slice_w - overlap

        self.prog["slice_l"].value = l
        self.prog["slice_r"].value = r

    # --------------------------------------------------------
    def paintGL(self):
        frame = self.sct.grab(self.src_geom)
        self.video_tex.write(frame.raw)
        self.vao.render(moderngl.TRIANGLE_STRIP)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--mode", required=True)
    args = parser.parse_args()

    app = QApplication(sys.argv)
    screens = QGuiApplication.screens()

    src = next(s for s in screens if s.name() == args.source)
    g = src.geometry()

    source_geom = {
        "left": g.x(),
        "top": g.y(),
        "width": g.width(),
        "height": g.height(),
    }

    n = len(args.targets)
    overlap_px = int(g.width() * 0.08)  # 8% 推奨

    windows = []
    for i, disp in enumerate(args.targets):
        screen = next(s for s in screens if s.name() == disp)
        display_id = get_virtual_id(disp)

        win = GLDisplayWindow(
            source_geom,
            screen,
            display_id,
            args.mode,
            i,
            n,
            overlap_px,
        )
        win.show()
        windows.append(win)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
