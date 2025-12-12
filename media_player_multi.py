# media_player_multi.py
# 360°映像のリアルタイム歪み補正・マルチディスプレイ出力プログラム
#
# このプログラムは、PyQtの QOpenGLWidget と ModernGL を使用し、
# 画面キャプチャ (mss) から GPUテクスチャ転送、シェーダーによる歪み補正までを
# 完全にGPUパイプライン上で処理することで、高フレームレートを実現しています。

import sys
import argparse
import numpy as np
import mss
import moderngl
import signal
import os

# PyQt5 GUIフレームワーク関連
from PyQt5.QtWidgets import QApplication, QOpenGLWidget
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtCore import QTimer, Qt

# ユーティリティ/ワーピングエンジン
from editor.grid_utils import load_points, log, get_virtual_id
# prepare_warp: 歪み補正マップ (map_x, map_y) を生成
# convert_maps_to_uv_texture_data: map_x, map_y をシェーダーで扱いやすいUVテクスチャデータに変換
from warp_engine import prepare_warp, convert_maps_to_uv_texture_data 


# --- QOpenGLWidget を継承した高性能描画ウィンドウ ---
class GLDisplayWindow(QOpenGLWidget):
    def __init__(self, source_screen, target_screen, mode, offset_x, virtual_size,
                 warp_info_all=None, fade_enabled=False):
        super().__init__()
        
        # 0. ウィンドウ設定
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_DeleteOnClose)
        
        # ターゲットディスプレイのジオメトリに合わせてウィンドウを設定
        g = target_screen.geometry()
        self.setGeometry(g.x(), g.y(), g.width(), g.height())

        # 1. メンバー変数設定
        self.source_screen = source_screen
        self.warp_info_all = warp_info_all # (map_x, map_y) のタプル
        self.offset_x = offset_x
        
        # 2. MSSの初期化 (キャプチャ範囲設定)
        self.sct = mss.mss()
        sg = source_screen.geometry()
        
        # キャプチャ領域辞書 (source_screen内での担当領域)
        self.monitor = {
            "top": sg.y(),
            "left": sg.x() + offset_x, # ソースディスプレイのX座標 + 自分の担当領域のオフセット
            "width": g.width(),        # 出力先の解像度と合わせる
            "height": g.height()
        }

        # 3. フレームレート制御用タイマー (QTimer.timeout -> update() -> paintGL() の流れ)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update) 
        self.timer.start(16) # 16ms間隔で更新 (約60fps)

    def initializeGL(self):
        """OpenGLの初期化：一度だけ呼ばれる (QOpenGLWidgetのライフサイクル)"""
        try:
            # ModernGL コンテキストの作成
            self.ctx = moderngl.create_context()
        except Exception as e:
            print(f"[FATAL ERROR] ModernGL context creation failed: {e}")
            sys.exit(1)
        
        # --- 1. 頂点データ (VBO / VAO) ---
        # 画面全体を覆う四角形 (TRIANGLE_STRIP)。データ構造: x, y, u, v
        vertices = np.array([
            -1.0, -1.0, 0.0, 1.0, # 左下 (GL:(-1,-1), UV:(0,1))
             1.0, -1.0, 1.0, 1.0, # 右下 (GL:(1,-1), UV:(1,1))
            -1.0,  1.0, 0.0, 0.0, # 左上 (GL:(-1,1), UV:(0,0))
             1.0,  1.0, 1.0, 0.0, # 右上 (GL:(1,1), UV:(1,0))
        ], dtype='f4')
        
        # --- 2. シェーダープログラム (GLSL) ---
        try:
            self.prog = self.ctx.program(
                vertex_shader="""
                    #version 330
                    in vec2 in_vert; // 頂点座標 (-1.0 to 1.0)
                    in vec2 in_text; // 基本テクスチャ座標 (0.0 to 1.0)
                    out vec2 v_text; // フラグメントシェーダーに渡すテクスチャ座標
                    void main() {
                        gl_Position = vec4(in_vert, 0.0, 1.0);
                        v_text = in_text;
                    }
                """,
                fragment_shader="""
                    #version 330
                    // binding 0: 画面キャプチャした元映像
                    uniform sampler2D original_tex; 
                    // binding 1: 歪み補正用UVマップ (R/Gチャンネルに x/y の参照座標を持つ)
                    uniform sampler2D warp_map_tex; 
                    
                    in vec2 v_text; // このフラグメントの画面座標に対応するUVマップ上の座標
                    out vec4 f_color;
                    
                    void main() {
                        // 1. UVマップテクスチャから「元画像上の参照すべきUV座標」を取得
                        // warp_map_tex の Rチャンネルが X (U)、Gチャンネルが Y (V)
                        vec2 source_uv = texture(warp_map_tex, v_text).rg;
                        
                        // 2. クリッピング (元画像の範囲外を参照している場合は黒にする)
                        if (source_uv.x < 0.0 || source_uv.x > 1.0 || 
                            source_uv.y < 0.0 || source_uv.y > 1.0) 
                        {
                            f_color = vec4(0.0, 0.0, 0.0, 1.0); // 黒
                        } else {
                            // 3. 元画像から補正済み座標の色を取得し、出力
                            f_color = texture(original_tex, source_uv);
                        }
                    }
                """
            )
        except Exception as e:
            print(f"\n[FATAL GLSL ERROR] シェーダーコンパイルまたはリンクに失敗しました:\n{e}")
            sys.exit(1)

        # VBO / VAO 作成
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [
            (self.vbo, '2f 2f', 'in_vert', 'in_text') # '2f': in_vert (xy), '2f': in_text (uv)
        ])

        # --- 3. テクスチャ作成 ---
        w = self.monitor["width"]
        h = self.monitor["height"]
        
        # 映像用テクスチャ (Binding 0: original_tex)
        # MSSはBGRA形式でデータを取得するため、4チャンネルで作成
        self.texture_video = self.ctx.texture((w, h), 4) 
        self.texture_video.swizzle = 'BGRA' # BGRA形式で受け取ったデータをRGBとして扱うよう設定
        self.texture_video.filter = (moderngl.LINEAR, moderngl.LINEAR) # 線形補間を有効に

        # 歪み補正マップ用テクスチャ (Binding 1: warp_map_tex)
        if self.warp_info_all:
            map_x, map_y = self.warp_info_all
            
            # map_x, map_y が numpy 配列であることを確認
            if not isinstance(map_x, np.ndarray) or not isinstance(map_y, np.ndarray):
                print(f"[FATAL ERROR] Warp map data is not a NumPy array! Type received: {type(map_x)}")
                sys.exit(1)
            
            # map_x, map_y をUV座標テクスチャデータ（R/Gチャンネル）に変換
            uv_data = convert_maps_to_uv_texture_data(map_x, map_y, w, h)
            
            # 2チャンネル (RG)、float32 型でテクスチャを作成・データを転送
            self.texture_warp = self.ctx.texture((w, h), 2, data=uv_data, dtype='f4')
            self.texture_warp.filter = (moderngl.NEAREST, moderngl.NEAREST) # マップは通常最近傍補間
        else:
            # 歪み補正マップがない場合は、恒等写像（歪みなし）のための空のテクスチャを作成（必須ではないが一応）
            self.texture_warp = self.ctx.texture((w, h), 2, dtype='f4') 

        # シェーダー内の uniform 変数にテクスチャのバインディング番号を設定
        self.prog['original_tex'].value = 0
        self.prog['warp_map_tex'].value = 1

    def paintGL(self):
        """毎フレーム呼ばれる描画処理 (QOpenGLWidgetのライフサイクル)"""
        # 1. 画面キャプチャ (CPU)
        # MSSの grab は非常に高速で、生バイト列 (BGRA形式) を取得
        sct_img = self.sct.grab(self.monitor)
        
        # 2. テクスチャ転送 (CPU -> GPU)
        # 画像変換処理なしに生バイト列をそのままGPUテクスチャに転送 (高速)
        self.texture_video.write(sct_img.raw)
        
        # 3. 描画実行 (GPU)
        self.texture_video.use(0) # 元画像をバインディング0にセット
        self.texture_warp.use(1)  # UVマップをバインディング1にセット
        self.vao.render(moderngl.TRIANGLE_STRIP) # シェーダーを実行して描画


def main():
    """メイン実行関数: コマンドライン引数の処理とウィンドウの起動を行う"""
    parser = argparse.ArgumentParser(description="リアルタイム歪み補正マルチディスプレイプレイヤー")
    parser.add_argument("--source", required=True, help="ソースディスプレイのPyQt名または仮想ID (例: D1)")
    parser.add_argument("--targets", nargs="+", required=True, help="出力先ディスプレイの仮想ID (例: D2 D3)")
    parser.add_argument("--mode", choices=["perspective", "warp_map"], default="perspective", help="補正方式")
    parser.add_argument("--blend", action="store_true", help="マルチターゲット時、アルファブレンドを有効にする (現在は未使用)")
    args = parser.parse_args()

    app = QApplication(sys.argv)

    # --- Ctrl+C (SIGINT)を有効化する処理 ---
    # PyQtのイベントループ中でもターミナルからのCtrl+Cを受け付けるようにするハック
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    timer = QTimer()
    timer.start(100) # 100msごとにPythonインタプリタに制御を戻す
    timer.timeout.connect(lambda: None) 
    # ------------------------------------

    # --- 入力された source / targets を内部ID (D1, D2, ...) に統一 ---
    # main.pyからD*形式で渡ってくることを想定し、両方の形式に対応させる
    src_vid = get_virtual_id(args.source)
    tgt_vids = [get_virtual_id(t) for t in args.targets]

    # QScreen を仮想ID (D1, D2, ...) をキーとして取得できる辞書を作成
    screens_by_name = {}
    for s in QGuiApplication.screens():
        vid = get_virtual_id(s.name())
        if vid:
            screens_by_name[vid] = s
    
    # 引数を仮想IDに更新
    args.source = src_vid
    args.targets = [t for t in tgt_vids if t and t in screens_by_name] # 見つからないターゲットは除外

    # ソースディスプレイの確認
    if args.source not in screens_by_name:
        print(f"❌ ソースディスプレイが見つかりません: {args.source}")
        sys.exit(1)

    source_screen = screens_by_name[args.source]

    # --- 仮想的な映像全体のサイズを計算 ---
    # total_width: 選択されたターゲットディスプレイの幅の合計
    total_width = sum(screens_by_name[n].geometry().width() for n in args.targets)
    # max_height: ターゲットディスプレイの最大高さ
    max_height = max((screens_by_name[n].geometry().height() for n in args.targets), 
                     default=source_screen.geometry().height())
    # virtual_size は現在のプログラムでは直接使用されていないが、今後の拡張のために保持
    # virtual_size = (total_width, max_height) 

    windows = []
    offset_x = 0 # ソースディスプレイ上のキャプチャ開始位置のオフセット

    # --- 各ターゲットディスプレイにウィンドウを起動 ---
    for name in args.targets:
        target_screen = screens_by_name[name]
        fade_enabled = args.blend and len(args.targets) > 1 # ブレンドフラグ
        
        # 歪み補正マップ (map_x, map_y) を事前に準備/ロード
        warp_info = prepare_warp(name, args.mode,
                                 (target_screen.geometry().width(), target_screen.geometry().height()),
                                 load_points_func=load_points, log_func=log)

        if warp_info is None:
            print(f"⚠️ {name} の warp 情報がありません。スキップします。")
            continue

        print(f"🎥 {args.source} → {name} 出力 (fade={fade_enabled})")

        # QOpenGLWidget を使用した描画ウィンドウを作成
        window = GLDisplayWindow(
            source_screen, target_screen, args.mode,
            offset_x, (total_width, max_height), # virtual_size
            warp_info_all=warp_info,
            fade_enabled=fade_enabled
        )
        window.show()
        windows.append(window)
        
        # 次のウィンドウのためにオフセットを更新
        offset_x += target_screen.geometry().width()

    if not windows:
        print("❌ 出力ディスプレイがありません。終了します。")
        sys.exit(1)

    # PyQtのイベントループを開始
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()