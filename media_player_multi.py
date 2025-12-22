# ======================================================
# media_player_multi.py (REVISED / OVERLAP + ALPHA)
# ======================================================
# 役割:
# - source ディスプレイを N 分割(短冊)
# - 各短冊を FHD に拡大
# - perspective / warp_map を切替
# - 左右 10% オーバーラップ + アルファブレンディング

import sys
import json
import argparse
import mss
import moderngl
import numpy as np
import cv2
from pathlib import Path

<<<<<<< HEAD
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtWidgets import QApplication, QOpenGLWidget

from editor.grid_utils import get_virtual_id

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"

OVERLAP_RATIO = 0.10  # 10% overlap
=======
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
>>>>>>> 97c68d26 (システム的には完全なる完成をしました。)


class GLDisplayWindow(QOpenGLWidget):
<<<<<<< HEAD
    def __init__(self, source, target, slice_geom, mode, warp_npz):
=======
    def __init__(self, source_screen, target_screen, mode,
                 warp_info_all=None,
                 source_geometry=None,
                 edit_display_name=None):
>>>>>>> 97c68d26 (システム的には完全なる完成をしました。)
        super().__init__()

        tg = target.geometry()
        self.setGeometry(tg)
        self.setWindowFlags(Qt.FramelessWindowHint)

        self.slice = slice_geom
        self.mode = mode

        self.full_w = source.geometry().width()
        self.full_h = source.geometry().height()

        self.sct = mss.mss()
        self.monitor = slice_geom

        self.warp_npz = warp_npz

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(16)

        self.edit_display_name = edit_display_name
        self.target_screen = target_screen

    def initializeGL(self):
        print("[INIT] OpenGL initialization start")
        print("[INIT] Warp map loading start")
        print("[INIT] Ready")

        self.ctx = moderngl.create_context()

        vertices = np.array([
            -1, -1, 0, 0,
             1, -1, 1, 0,
            -1,  1, 0, 1,
             1,  1, 1, 1,
        ], dtype='f4')

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_pos;
                in vec2 in_uv;
                out vec2 v_uv;
                void main(){ gl_Position=vec4(in_pos,0,1); v_uv=in_uv; }
            ''',
            fragment_shader='''
                #version 330
                in vec2 v_uv;
                out vec4 fragColor;

                uniform sampler2D video_tex;
                uniform sampler2D warp_tex;
                uniform float alpha_l;
                uniform float alpha_r;
                uniform int mode;

                void main(){
                    vec2 uv = v_uv;
                    if(mode==1){ uv = texture(warp_tex, v_uv).rg; }

                    float a = 1.0;
                    if(v_uv.x < alpha_l) a = v_uv.x/alpha_l;
                    if(v_uv.x > 1.0-alpha_r) a = (1.0-v_uv.x)/alpha_r;

                    fragColor = vec4(texture(video_tex, uv).rgb, a);
                }
            '''
        )

        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog, [(self.vbo, '2f 2f', 'in_pos', 'in_uv')]
        )

        self.video_tex = self.ctx.texture((self.full_w, self.full_h), 4)
        self.video_tex.swizzle = 'BGRA'

<<<<<<< HEAD
        if self.mode == 'map':
            mx = self.warp_npz['map_x']
            my = self.warp_npz['map_y']
            uv = np.dstack([mx, my]).astype('f4')
            self.warp_tex = self.ctx.texture((self.full_w, self.full_h), 2, uv.tobytes(), dtype='f4')
            self.warp_tex.use(1)

        self.prog['alpha_l'].value = OVERLAP_RATIO
        self.prog['alpha_r'].value = OVERLAP_RATIO
        self.prog['mode'].value = 1 if self.mode=='map' else 0
=======
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
>>>>>>> 97c68d26 (システム的には完全なる完成をしました。)

    def paintGL(self):
        frame = np.array(self.sct.grab(self.monitor))
        frame = cv2.resize(frame, (self.full_w, self.full_h))
        self.video_tex.write(frame.tobytes())
        self.video_tex.use(0)
        self.vao.render(moderngl.TRIANGLE_STRIP)

<<<<<<< HEAD

# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--mode', required=True)
    parser.add_argument('--targets', nargs='+', required=True)
    args = parser.parse_args()

    app = QApplication(sys.argv)
    screens = {s.name(): s for s in QGuiApplication.screens()}

    src = screens[args.source]
    full_w = src.geometry().width()
    full_h = src.geometry().height()

    wins = []
    n = len(args.targets)

    for i, t in enumerate(args.targets):
        tgt = screens[t]
        overlap = int(full_w * OVERLAP_RATIO)
        x0 = max(0, int(i*full_w/n - overlap))
        w0 = int(full_w/n + overlap*2)

        slice_geom = {
            'top': 0,
            'left': x0,
            'width': w0,
            'height': full_h
        }

        sim = f'ScreenSimulatorSet_{i+1}'
        warp_npz = np.load(CONFIG_DIR/'warp_cache'/f'{sim}_map_{full_w}x{full_h}.npz')

        win = GLDisplayWindow(src, tgt, slice_geom, args.mode, warp_npz)
        win.show()
        wins.append(win)

    sys.exit(app.exec_())


if __name__=='__main__':
=======
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
>>>>>>> 97c68d26 (システム的には完全なる完成をしました。)
    main()
