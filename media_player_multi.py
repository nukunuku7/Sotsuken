# media_player_multi.py

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

        # パラメータ保存
        self.source_screen = source_screen
        self.warp_info_all = warp_info_all

        # スライスジオメトリ情報
        self.slice_index = source_geometry.get("index", 0)
        self.slice_count = source_geometry.get("count", 1)
        self.overlap_px = source_geometry.get("overlap", 0)
        self.enable_blend = self.slice_count > 1

        slice_w = source_geometry.get("w", 1)
        self.slice_valid_left = self.overlap_px / slice_w
        self.slice_valid_right = 1.0 - self.slice_valid_left
                
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

        # ブレンド有効化
        if self.enable_blend:
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (
                moderngl.SRC_ALPHA,
                moderngl.ONE_MINUS_SRC_ALPHA,
            )

        # シェーダー作成
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

                uniform float slice_left;
                uniform float slice_right;
                uniform int enable_blend;

                in vec2 v_uv;
                out vec4 fragColor;

                void main() {
                    vec2 warped_uv = texture(warp_uv_tex, v_uv).rg;
                    warped_uv = clamp(warped_uv, 0.0, 1.0);
                    vec4 color = texture(original_tex, warped_uv);

                    if (enable_blend == 1) {
                        float alpha = 1.0;

                        // ★ 短冊の左フェード
                        if (warped_uv.x < slice_left) {
                            alpha = smoothstep(0.0, slice_left, warped_uv.x);
                        }
                        // ★ 短冊の右フェード
                        else if (warped_uv.x > slice_right) {
                            alpha = smoothstep(1.0, slice_right, warped_uv.x);
                        }

                        color.a *= alpha;
                    }

                    fragColor = color;
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

        body_w = cap_w - self.overlap_px * 2   # 元の短冊幅
        
        # 映像用テクスチャ (Binding 0)
        self.texture_video = self.ctx.texture((cap_w, cap_h), 4) # BGRA=4ch
        self.texture_video.swizzle = 'BGRA' # BGRA -> RGBへスウィズル(並び替え)
        
        # 歪み補正マップ用テクスチャ (Binding 1)
        # warp_engine から map_x, map_y を取得済みと仮定
        if self.warp_info_all is not None:
            map_x, map_y = self.warp_info_all

            target_w = self.width()
            target_h = self.height()

            cap_w = self.monitor["width"]
            cap_h = self.monitor["height"]
            body_w = cap_w - self.overlap_px * 2

            map_x = map_x.astype(np.float32)
            map_y = map_y.astype(np.float32)

            # target → short strip
            map_x = map_x * (body_w / target_w)
            map_y = map_y * (cap_h / target_h)

            # overlap
            map_x = (map_x + self.overlap_px) / cap_w
            map_y = map_y / cap_h

            # map_x, map_y は short-strip 解像度
            # → 出力解像度にリサンプルする
            uv_data = cv2.resize(
                np.dstack([map_x, map_y]),
                (target_w, target_h),
                interpolation=cv2.INTER_LINEAR
            ).astype("f4")

            print(
                f"[DEBUG] {self.windowTitle() or 'proj'} "
                f"map_x min/max = {map_x.min()} / {map_x.max()}, "
                f"source_w = {self.monitor['width']}"
            )

        else:
            # warp_map が無い場合：恒等UVを自前生成
            cap_w = self.monitor["width"]
            cap_h = self.monitor["height"]

            xs = np.linspace(0.0, 1.0, cap_w, dtype=np.float32)
            ys = np.linspace(0.0, 1.0, cap_h, dtype=np.float32)

            u, v = np.meshgrid(xs, ys)
            uv_data = np.dstack([u, v]).astype("f4")

            print(
                f"[DEBUG] {self.windowTitle() or 'proj'} "
                f"identity UV map, source_w = {self.monitor['width']}"
            )

        warp_h, warp_w = uv_data.shape[:2]

        self.texture_warp = self.ctx.texture(
            (warp_w, warp_h),
            2,
            data=uv_data,
            dtype='f4'
        )

        # シェーダーにテクスチャ番号を教える
        self.prog['original_tex'].value = 0
        self.prog['warp_uv_tex'].value = 1
        self.prog["slice_left"].value = self.slice_valid_left
        self.prog["slice_right"].value = self.slice_valid_right
        self.prog["enable_blend"].value = 1 if self.enable_blend else 0

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
    overlap_ratio = 0.1 if slice_count > 1 else 0.0
    slice_w = sg.width() // slice_count
    overlap_px = int(slice_w * overlap_ratio)
    slice_h = sg.height()

    # 各ターゲットディスプレイごとにウィンドウを作成
    for proj_index, name in enumerate(args.targets):
        slice_x = src_base_x + slice_w * proj_index
        slice_y = src_base_y

        # オーバーラップ分を拡張
        cap_x = slice_x - overlap_px
        cap_w = slice_w + overlap_px * 2

        # 画面外に出ないようにクランプ
        cap_x = max(src_base_x, cap_x)

        # 右端もはみ出さないように
        max_x = src_base_x + sg.width() - cap_w
        cap_x = min(cap_x, max_x)

        slice_geometry = {
            "x": cap_x,
            "y": slice_y,
            "w": cap_w,
            "h": slice_h,
            "index": proj_index,
            "count": slice_count,
            "overlap": overlap_px,
        }

        if name not in screens_by_name:
            print(f"⚠️ ターゲットディスプレイが見つかりません: {name}")
            continue

        target_screen = screens_by_name[name]

        # ★★★ ここが最重要修正 ★★★
        warp_info = prepare_warp(
            name,
            args.mode,
            (slice_geometry["w"],
             slice_geometry["h"]),  # 1920x1080
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
