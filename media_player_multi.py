# media_player_multi.py
import sys
import argparse
import mss
import moderngl
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtWidgets import QOpenGLWidget, QApplication

from editor.grid_utils import get_virtual_id, log
from warp_engine import prepare_warp, convert_maps_to_uv_texture_data


# ------------------------------------------------------------
# GL Window
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
        self.sct = mss.mss()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(16)

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
                    vec2 warped_uv = texture(warp_tex, v_uv).rg;

                    if (warped_uv.x < 0.0 || warped_uv.x > 1.0 ||
                        warped_uv.y < 0.0 || warped_uv.y > 1.0) {
                        fragColor = vec4(0.0);
                        return;
                    }

                    vec4 col = texture(video_tex, warped_uv);

                    float a = 1.0;
                    if (v_uv.x < slice_l)
                        a = smoothstep(0.0, slice_l, v_uv.x);
                    else if (v_uv.x > slice_r)
                        a = smoothstep(1.0, slice_r, v_uv.x);

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
        w = self.src_geom["w"]
        h = self.src_geom["h"]
        self.video_tex = self.ctx.texture((w, h), 4)
        self.video_tex.swizzle = "BGRA"

        # warp map
        map_pair = prepare_warp(
            display_name=self.display_id,
            mode=self.mode,
            src_size=(w, h),
            log_func=log,
        )
        if map_pair is None:
            raise RuntimeError("Grid data missing")

        map_x, map_y = map_pair
        uv_bytes = convert_maps_to_uv_texture_data(map_x, map_y, w, h)
        self.warp_tex = self.ctx.texture((w, h), 2, uv_bytes, dtype="f4")

        self.video_tex.use(0)
        self.warp_tex.use(1)
        self.prog["video_tex"].value = 0
        self.prog["warp_tex"].value = 1

        slice_w = 1.0 / self.slice_count
        overlap = self.overlap_px / w

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
    geom = src.geometry()

    source_geom = {
        "left": geom.x(),
        "top": geom.y(),
        "w": geom.width(),
        "h": geom.height(),
    }

    windows = []
    n = len(args.targets)
    overlap_px = int(geom.width() * 0.1)

    for i, disp in enumerate(args.targets):
        screen = next(s for s in screens if s.name() == disp)
        display_id = get_virtual_id(disp)

        win = GLDisplayWindow(
            source_geom=source_geom,
            target_screen=screen,
            display_id=display_id,
            mode=args.mode,
            slice_index=i,
            slice_count=n,
            overlap_px=overlap_px,
        )
        win.show()
        windows.append(win)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
