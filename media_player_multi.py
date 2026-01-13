#　media_player_multi.py
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
from warp_engine import prepare_warp


class GLDisplayWindow(QOpenGLWidget):
    def __init__(self, source_screen, target_screen, mode,
                 warp_info_all=None,
                 source_geometry=None):
        super().__init__()

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_DeleteOnClose)

        g = target_screen.geometry()
        self.setFixedSize(g.width(), g.height())
        self.move(g.x(), g.y())

        self.source_geometry = source_geometry
        self.source_screen = source_screen
        self.warp_info_all = warp_info_all

        # ----------------------------
        # slice 情報
        # ----------------------------
        self.slice_index = source_geometry["index"]
        self.slice_count = source_geometry["count"]

        self.overlap_l_px = source_geometry["overlap_left"]
        self.overlap_r_px = source_geometry["overlap_right"]
        self.body_w_px = source_geometry["body_width"]
        self.cap_w_px = self.overlap_l_px + self.body_w_px + self.overlap_r_px

        self.enable_blend = self.slice_count > 1

        # ----------------------------
        # MSS キャプチャ領域
        # ----------------------------
        self.sct = mss.mss()
        # キャプチャは body + overlap
        self.monitor = {
            "top": source_geometry["y"],
            "left": source_geometry["x"],
            "width": self.cap_w_px,
            "height": source_geometry["h"],
        }

        # ----------------------------
        # キャプチャ正規化
        # ----------------------------
        self.body_to_cap = self.body_w_px / self.cap_w_px
        self.cap_offset = self.overlap_l_px / self.cap_w_px
        self.overlap_norm = self.overlap_l_px / self.body_w_px if self.body_w_px > 0 else 0.0

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(16)

    def initializeGL(self):
        self.ctx = moderngl.create_context()

        # ======================================
        # fullscreen quad
        vertices = np.array([
            -1.0, -1.0, 0.0, 1.0,
            1.0, -1.0, 1.0, 1.0,
            -1.0,  1.0, 0.0, 0.0,
            1.0,  1.0, 1.0, 0.0,
        ], dtype="f4")

        if self.enable_blend:
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (
                moderngl.ONE,
                moderngl.ONE_MINUS_SRC_ALPHA,
            )

        # ======================================
        # Shader
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
                #version 330 core

                uniform sampler2D original_tex;
                uniform sampler2D warp_uv_tex;
                uniform sampler2D warp_mask_tex;

                uniform float cap_offset;
                uniform float body_to_cap;
                uniform float overlap_norm;

                uniform int fade_left;
                uniform int fade_right;
                uniform int enable_blend;

                in vec2 v_uv;
                out vec4 fragColor;

                void main() {
                    float mask = texture(warp_mask_tex, v_uv).r;
                    if (mask < 0.5)
                        discard;

                    vec2 warp_uv = texture(warp_uv_tex, v_uv).rg;

                    // body → cap
                    vec2 cap_uv;
                    cap_uv.x = warp_uv.x * body_to_cap + cap_offset;
                    cap_uv.y = warp_uv.y;

                    vec4 color = texture(original_tex, cap_uv);

                    if (enable_blend == 1 && overlap_norm > 0.0) {
                        float alpha = 1.0;

                        if (fade_left == 1 && warp_uv.x < overlap_norm) {
                            alpha = warp_uv.x / overlap_norm;
                        }

                        if (fade_right == 1 && warp_uv.x > 1.0 - overlap_norm) {
                            alpha = (1.0 - warp_uv.x) / overlap_norm;
                        }

                        alpha = clamp(alpha, 0.0, 1.0);
                        color.rgb *= alpha;
                        color.a = alpha;

                        if (color.a <= 0.0001)
                            discard;
                    }

                    fragColor = color;
                }
            """
        )

        # ======================================
        # VAO
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, "2f 2f", "in_vert", "in_text")]
        )

        # ======================================
        # Video texture (cap size)
        cap_w = self.cap_w_px
        cap_h = self.source_geometry["h"]

        self.texture_video = self.ctx.texture((cap_w, cap_h), 4)
        self.texture_video.swizzle = "BGRA"

        # ======================================
        # Warp UV texture (body size → screen size)
        map_x, map_y, valid_mask = self.warp_info_all
        tw, th = self.width(), self.height()

        # resize to window framebuffer
        map_x = cv2.resize(map_x, (tw, th), interpolation=cv2.INTER_LINEAR)
        map_y = cv2.resize(map_y, (tw, th), interpolation=cv2.INTER_LINEAR)
        uv_data = np.dstack([map_x, map_y]).astype("f4")

        self.texture_warp = self.ctx.texture(
            (uv_data.shape[1], uv_data.shape[0]),
            2,
            data=uv_data,
            dtype="f4"
        )

        valid_mask = cv2.resize(valid_mask, (tw, th), interpolation=cv2.INTER_NEAREST)
        self.texture_mask = self.ctx.texture(
            (tw, th),
            1,
            data=valid_mask.astype("f4"),
            dtype="f4"
        )

        # ======================================
        # Uniforms
        self.prog["original_tex"].value = 0
        self.prog["warp_uv_tex"].value = 1
        self.prog["warp_mask_tex"].value = 2

        self.prog["cap_offset"].value  = self.cap_offset
        self.prog["body_to_cap"].value = self.body_to_cap
        self.prog["overlap_norm"].value = self.overlap_norm

        self.prog["fade_left"].value  = 1 if self.slice_index > 0 else 0
        self.prog["fade_right"].value = 1 if self.slice_index < self.slice_count - 1 else 0
        self.prog["enable_blend"].value = 1 if self.enable_blend else 0

    def resizeGL(self, w, h):
        dpr = self.devicePixelRatioF()
        log(f"[Qt] resizeGL logical={w}x{h}, framebuffer={int(w*dpr)}x{int(h*dpr)}")

    def paintGL(self):
        """毎フレーム描画"""

        dpr = self.devicePixelRatioF()
        w = int(self.width() * dpr)
        h = int(self.height() * dpr)
        self.ctx.viewport = (0, 0, w, h)

        # ----------------------------
        # MSSキャプチャ (body + overlap)
        # ----------------------------
        sct_img = self.sct.grab(self.monitor)
        self.texture_video.write(sct_img.raw)

        # ----------------------------
        # 描画
        # ----------------------------
        self.texture_video.use(0)
        self.texture_warp.use(1)
        self.texture_mask.use(2)
        self.vao.render(moderngl.TRIANGLE_STRIP)

        if not hasattr(self, "_once"):
            self._once = True
            print("[DEBUG MSS] monitor =", self.monitor, "size =", sct_img.size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--mode", choices=["perspective", "warp_map"], default="perspective")
    parser.add_argument("--blend", action="store_true", help="Enable alpha blending")
    args = parser.parse_args()

    QApplication.setAttribute(Qt.AA_DisableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_Use96Dpi)

    app = QApplication(sys.argv)

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    timer = QTimer()
    timer.start(100)
    timer.timeout.connect(lambda: None)

    # --- 仮想ID
    src_vid = get_virtual_id(args.source)
    if not src_vid:
        print(f"❌ ソース {args.source} の内部ID変換に失敗")
        sys.exit(1)

    tgt_vids = [get_virtual_id(t) for t in args.targets if get_virtual_id(t)]
    args.source = src_vid
    args.targets = tgt_vids

    screens_by_name = {}
    for s in QGuiApplication.screens():
        vid = get_virtual_id(s.name())
        if vid:
            screens_by_name[vid] = s

    if args.source not in screens_by_name:
        print(f"❌ ソースディスプレイが見つかりません: {args.source}")
        sys.exit(1)

    source_screen = screens_by_name[args.source]
    sg = source_screen.geometry()

    src_base_x = sg.x()
    src_base_y = sg.y()

    slice_count = len(args.targets)
    slice_w = sg.width() // slice_count
    slice_h = sg.height()

    overlap_ratio = 0.1 if slice_count > 1 else 0.0
    overlap_px = int(slice_w * overlap_ratio)

    windows = []

    for proj_index, name in enumerate(args.targets):
        body_x = src_base_x + slice_w * proj_index

        # -------------------------------
        # overlap 計算（キャプチャ専用）
        # -------------------------------
        if slice_count == 1:
            overlap_l = overlap_r = 0
        else:
            if proj_index == 0:
                overlap_l = 0
                overlap_r = overlap_px
            elif proj_index == slice_count - 1:
                overlap_l = overlap_px
                overlap_r = 0
            else:
                overlap_l = overlap_px
                overlap_r = overlap_px

        cap_x = body_x - overlap_l
        cap_w = slice_w + overlap_l + overlap_r

        # clamp
        cap_x = max(src_base_x, cap_x)
        max_x = src_base_x + sg.width() - cap_w
        cap_x = min(cap_x, max_x)

        slice_geometry = {
            "x": cap_x,
            "y": src_base_y,
            "w": cap_w,
            "h": slice_h,
            "index": proj_index,
            "count": slice_count,
            "overlap_left": overlap_l,
            "overlap_right": overlap_r,
            "body_width": slice_w,
        }

        if name not in screens_by_name:
            print(f"⚠️ ターゲットディスプレイが見つかりません: {name}")
            continue

        target_screen = screens_by_name[name]

        warp_info = prepare_warp(
            name,
            args.mode,
            (slice_w, slice_h),
            load_points_func=load_points,
            log_func=log
        )

        print(f"🎥 {args.source} → {name} 出力")

        window = GLDisplayWindow(
            source_screen,
            target_screen,
            args.mode,
            warp_info_all=warp_info,
            source_geometry=slice_geometry
        )
        window.show()
        windows.append(window)

    if not windows:
        print("❌ 出力ディスプレイがありません。終了します。")
        sys.exit(1)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
