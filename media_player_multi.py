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

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtWidgets import QApplication, QOpenGLWidget

from editor.grid_utils import get_virtual_id

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"

OVERLAP_RATIO = 0.10  # 10% overlap


class GLDisplayWindow(QOpenGLWidget):
    def __init__(self, source, target, slice_geom, mode, warp_npz):
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

    def initializeGL(self):
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

        if self.mode == 'map':
            mx = self.warp_npz['map_x']
            my = self.warp_npz['map_y']
            uv = np.dstack([mx, my]).astype('f4')
            self.warp_tex = self.ctx.texture((self.full_w, self.full_h), 2, uv.tobytes(), dtype='f4')
            self.warp_tex.use(1)

        self.prog['alpha_l'].value = OVERLAP_RATIO
        self.prog['alpha_r'].value = OVERLAP_RATIO
        self.prog['mode'].value = 1 if self.mode=='map' else 0

    def paintGL(self):
        frame = np.array(self.sct.grab(self.monitor))
        frame = cv2.resize(frame, (self.full_w, self.full_h))
        self.video_tex.write(frame.tobytes())
        self.video_tex.use(0)
        self.vao.render(moderngl.TRIANGLE_STRIP)


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
    main()
