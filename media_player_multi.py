# media_player_multi.py

import sys
import mss
import signal
import argparse
import moderngl
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtWidgets import QOpenGLWidget, QApplication

from editor.grid_utils import load_points, log, get_virtual_id
from warp_engine import prepare_warp, convert_maps_to_uv_texture_data


class GLDisplayWindow(QOpenGLWidget):
    def __init__(self, source_screen, target_screen, mode,
                 warp_info_all=None,
                 source_geometry=None):
        super().__init__()

        # ウィンドウ設定
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_DeleteOnClose)
        
        # スクリーン配置
        g = target_screen.geometry()
        self.setFixedSize(g.width(), g.height())
        self.move(g.x(), g.y())

        self.source_screen = source_screen
        self.warp_info_all = warp_info_all
        
        # MSSの初期化 (キャプチャ範囲設定)
        self.sct = mss.mss()
        # source_screen の座標を取得
        sg = source_geometry

        # MSS用のキャプチャ領域辞書
        self.monitor = {
            "top": sg["y"],
            "left": sg["x"],
            "width": sg["w"],
            "height": sg["h"],
        }

        # フレームレート制御用タイマー (60FPS目標)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update) # update() が paintGL() を呼ぶ
        self.timer.start(16) # 約60fps

    def initializeGL(self):
        """OpenGLの初期化：一度だけ呼ばれる"""
        self.ctx = moderngl.create_context()

        # === GPU 情報を取得して表示 ==========================
        try:
            vendor = self.ctx.info["GL_VENDOR"]
            renderer = self.ctx.info["GL_RENDERER"]
            version = self.ctx.info["GL_VERSION"]
            log(f"🟢 GPU 検出: {renderer} ({vendor})")
            log(f"    OpenGL Version: {version}")
        except Exception as e:
            log(f"⚠️ GPU 情報の取得に失敗しました: {e}")
        # =====================================================

        # 1. 頂点データ（画面全体を覆う四角形）
        # x, y, u, v
        vertices = np.array([
            -1.0, -1.0, 0.0, 1.0, # 左下 (画像座標系では左上に対応させるためVを反転等の調整が必要かも)
             1.0, -1.0, 1.0, 1.0, # 右下
            -1.0,  1.0, 0.0, 0.0, # 左上
             1.0,  1.0, 1.0, 0.0, # 右上
        ], dtype='f4')
        
        try:
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

                in vec2 v_uv;
                out vec4 fragColor;

                void main() {
                    vec2 warped_uv = texture(warp_uv_tex, v_uv).rg;
                    fragColor = texture(original_tex, warped_uv);
                }

                """
            )
        except Exception as e:
            # ★★★ 強制的にエラーメッセージを出力 ★★★
            print(f"\n[FATAL GLSL ERROR] シェーダーコンパイルまたはリンクに失敗しました:\n{e}")
            import sys
            # エラーを表示させるためにプロセスを強制終了
            sys.exit(1)
            
        # VBO / VAO 作成
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [
            (self.vbo, '2f 2f', 'in_vert', 'in_text')
        ])

        # 2. テクスチャ作成
        cap_w = self.monitor["width"]
        cap_h = self.monitor["height"]
        
        # 映像用テクスチャ (Binding 0)
        self.texture_video = self.ctx.texture((cap_w, cap_h), 4) # BGRA=4ch
        self.texture_video.swizzle = 'BGRA' # BGRA -> RGBへスウィズル(並び替え)
        
        # 歪み補正マップ用テクスチャ (Binding 1)
        # warp_engine から map_x, map_y を取得済みと仮定
        if self.warp_info_all:
            map_x, map_y = self.warp_info_all

            # ★ 短冊サイズ
            sw = self.monitor["width"]
            sh = self.monitor["height"]

            # ★ 念のため float 化
            map_x = map_x.astype(np.float32)
            map_y = map_y.astype(np.float32)

            # ★ ここが決定打：短冊ローカルに正規化
            scale_x = sw / self.width()   # 640 / 1920 = 1/3
            scale_y = sh / self.height()  # 1080 / 1080 = 1

            map_x = (map_x / sw) * scale_x
            map_y = (map_y / sh) * scale_y

            if not isinstance(map_x, np.ndarray) or not isinstance(map_y, np.ndarray):
                 print(f"[FATAL ERROR] Warp map data is not a NumPy array! Type received: {type(map_x)}")
                 import sys
                 # ログを出力して終了し、原因を明確にする
                 sys.exit(1)
            
            # ★ここで手順2で作った変換関数を使う
            uv_data = np.dstack([map_x, map_y]).astype("f4")

            warp_h, warp_w = map_x.shape  # map_x は (H, W)

            self.texture_warp = self.ctx.texture(
                (warp_w, warp_h),
                2,
                data=uv_data,
                dtype='f4'
            )
        else:
            # マップがない場合は恒等写像（歪みなし）を作る等の処理
            uv_data = convert_maps_to_uv_texture_data(
                map_x,
                map_y,
                cap_w,     # ← source 幅（短冊）
                cap_h      # ← source 高さ
            )
        # シェーダーにテクスチャ番号を教える
        self.prog['original_tex'].value = 0
        self.prog['warp_uv_tex'].value = 1

        print(
            f"[DEBUG] {self.windowTitle() or 'proj'} "
            f"map_x min/max = {map_x.min()} / {map_x.max()}, "
            f"source_w = {self.monitor['width']}"
        )

    def resizeGL(self, w, h):
        # ★ これが「GPUが実際に描くサイズ」
        dpr = self.devicePixelRatioF()
        log(f"[Qt] resizeGL logical={w}x{h}, framebuffer={int(w*dpr)}x{int(h*dpr)}")

    def paintGL(self):
        """毎フレーム呼ばれる描画処理"""

        # ★ Qt が viewport を上書きした直後なので、ここで再設定する
        dpr = self.devicePixelRatioF()

        w = int(self.width() * dpr)
        h = int(self.height() * dpr)

        self.ctx.viewport = (0, 0, w, h)  # 1920x1080

        # 1. 画面キャプチャ (CPU)
        # MSSの grab は非常に高速ですが、ここのバイナリ取得だけが唯一のCPUコストです
        sct_img = self.sct.grab(self.monitor)
        
        # 2. テクスチャ転送 (CPU -> GPU)
        # 画像変換(opencv等)は一切せず、生バイト列をそのままGPUに投げ込む
        self.texture_video.write(sct_img.raw)
        
        # 3. 描画実行 (GPU)
        self.texture_video.use(0)
        self.texture_warp.use(1)
        self.vao.render(moderngl.TRIANGLE_STRIP)

        if not hasattr(self, "_once"):
            self._once = True
            print("[DEBUG MSS]")
            print(" monitor =", self.monitor)
            print(" sct_img.size =", sct_img.size)

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

    # --- Ctrl+C (SIGINT)を有効化する処理 ★ここを追加★ ---
    # 1. SIGINT のハンドラをデフォルトに戻す
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    # 2. PyQtのイベントループが実行中でも、Pythonがシグナルをチェックするように
    # わずかな間隔で空の QTimer を発火させる（Pythonインタプリタに制御を戻すためのハック）
    timer = QTimer()
    timer.start(100) # 100msごとにチェック
    timer.timeout.connect(lambda: None) 
    # ---------------------------------------------------

    # --- 入力された source / targets を内部IDに統一 ---
    src_vid = get_virtual_id(args.source)
    tgt_vids = [get_virtual_id(t) for t in args.targets]

    if not src_vid:
        print(f"❌ ソース {args.source} の内部ID変換に失敗")
        sys.exit(1)

    args.source = src_vid
    args.targets = [get_virtual_id(t) for t in args.targets if get_virtual_id(t)]


    # --- QScreen を名前別に取得（QScreen.name() と仮想ID の両方をキーにする） ---
    screens_by_name = {}
    # 追加で仮想 ID (D1, D2, ...) もキーにしておく（main.py から D* が渡されても解決できるように）
    for s in QGuiApplication.screens():
        vid = get_virtual_id(s.name())
        if vid:
            screens_by_name[vid] = s
    # -------------------------------------------------------------------------

    if args.source not in screens_by_name:
        print(f"❌ ソースディスプレイが見つかりません: {args.source}")
        sys.exit(1)

    source_screen = screens_by_name[args.source]

    # total_width は targets のうち見つかったスクリーン幅の合計
    total_width = sum(screens_by_name[n].geometry().width() for n in args.targets if n in screens_by_name)
    # max_height は利用可能なスクリーン全体の最大高さ（または targets の最大高さでも良い）
    max_height = max((s.geometry().height() for s in screens_by_name.values()), default= source_screen.geometry().height())
    virtual_size = (total_width, max_height)

    windows = []
    offset_x = 0

    # まず source のジオメトリを取得しておく
    source_screen = screens_by_name[args.source]
    sg = source_screen.geometry()   # ★ source geometry はここで一度だけ

    # ★ 追加：source の左上（仮想デスクトップ基準）
    src_base_x = sg.x()
    src_base_y = sg.y()

    slice_count = len(args.targets)
    slice_w = sg.width() // slice_count
    slice_h = sg.height()

    # 各ターゲットディスプレイごとにウィンドウを作成
    for proj_index, name in enumerate(args.targets):
        slice_x = src_base_x + slice_w * proj_index
        slice_y = src_base_y

        slice_geometry = {
            "x": slice_x,
            "y": slice_y,
            "w": slice_w,
            "h": slice_h,
        }

        if name not in screens_by_name:
            print(f"⚠️ ターゲットディスプレイが見つかりません: {name}")
            continue

        target_screen = screens_by_name[name]

        # ★★★ ここが最重要修正 ★★★
        warp_info = prepare_warp(
            name,
            args.mode,
            (target_screen.geometry().width(),
             target_screen.geometry().height()),  # 1920x1080
            load_points_func=load_points,
            log_func=log
        )

        print(f"🎥 {args.source} → {name} 出力")

        window = GLDisplayWindow(
            source_screen,
            target_screen,
            args.mode,
            warp_info_all=warp_info,
            source_geometry=slice_geometry # ★ source_geometry を渡す
        )
        window.show()
        windows.append(window)
        offset_x += target_screen.geometry().width()

    if not windows:
        print("❌ 出力ディスプレイがありません。終了します。")
        sys.exit(1)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
