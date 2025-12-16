import sys
import argparse
import cv2
import numpy as np
import mss
from PyQt5.QtWidgets import QApplication, QLabel, QWidget
from PyQt5.QtGui import QImage, QPixmap, QGuiApplication
from PyQt5.QtCore import QTimer, Qt

from editor.grid_utils import load_points, log, get_virtual_id
from warp_engine import warp_image, prepare_warp

# === GPU 自動検出 ==================================================
try:
    # cv2.cuda が使えるか確認
    import cv2.cuda as cuda
    GPU_AVAILABLE = cuda.getCudaEnabledDeviceCount() > 0
    if GPU_AVAILABLE:
        log("◎ CUDA GPU が検出されました。GPU処理を使用します。")
    else:
        log("△ GPU は利用できません。CPU処理になります。")
except Exception:
    GPU_AVAILABLE = False
    log("△ CUDA が利用できないため CPUモードで動作します。")
# ===================================================================


<<<<<<< HEAD
class DisplayWindow(QWidget):
    def __init__(self, source_screen, target_screen, mode, offset_x, virtual_size,
                 warp_info_all=None, fade_enabled=False):
=======
# ============================================================
# OpenGL Window
# ============================================================
class GLDisplayWindow(QOpenGLWidget):
    def __init__(self, source_screen, target_screen,
                 slice_offset_x, slice_size, warp_info):
>>>>>>> 941fe4942b97dcdad64f5aa145809e1d66a430b8
        super().__init__()
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_DeleteOnClose)

<<<<<<< HEAD
        self.source_screen = source_screen
        self.target_screen = target_screen
        self.mode = mode
        self.offset_x = offset_x
        self.virtual_size = virtual_size
        self.fade_enabled = fade_enabled
        self.warp_info = warp_info_all
        self.use_gpu = GPU_AVAILABLE  # === GPUフラグ ===

        geom_tgt = target_screen.geometry()
        self.setGeometry(geom_tgt)
        self.label = QLabel(self)
        self.label.setGeometry(0, 0, geom_tgt.width(), geom_tgt.height())

        # キャプチャ設定
        self.sct = mss.mss()
        geom_src = source_screen.geometry()
        self.mon = {
            "left": geom_src.x(),
            "top": geom_src.y(),
            "width": geom_src.width(),
            "height": geom_src.height()
=======
        # --- 表示先（プロジェクター）全画面
        g = target_screen.geometry()
        self.setGeometry(g.x(), g.y(), g.width(), g.height())

        self.source_screen = source_screen
        self.slice_offset_x = slice_offset_x   # ★ 将来用（現在は未使用）
        self.slice_size = slice_size
        self.warp_info = warp_info

        self.target_width = g.width()
        self.target_height = g.height()

        # --- MSS（編集画面全体をキャプチャ）
        self.sct = mss.mss()
        sg = source_screen.geometry()
        self.monitor = {
            "top": sg.y(),
            "left": sg.x(),
            "width": sg.width(),
            "height": sg.height(),
>>>>>>> 941fe4942b97dcdad64f5aa145809e1d66a430b8
        }

        # warp 情報（ターゲット名は QScreen.name() か、media 側で仮想IDを解決して渡される）
        vid = get_virtual_id(target_screen.name())
        points_local = load_points(vid, mode)
        if not points_local:
            log(f"[WARN] グリッドが存在しないためスキップ: {target_screen.name()}")
            self.warp_info = None
        else:
            total_w, total_h = virtual_size
            adjusted_points = []
            for p in points_local:
                x_adj = p[0] + self.offset_x
                y_adj = p[1]
                adjusted_points.append([x_adj, y_adj])

            self.warp_info = prepare_warp(
                display_name=vid,
                mode=self.mode,
                src_size=(geom_tgt.width(), geom_tgt.height()),
                load_points_func=lambda *_: adjusted_points,
                log_func=log
            )

        # === 60fps に変更 =======
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(16)  # 16ms = 60fps
        # ========================

<<<<<<< HEAD
        self.showFullScreen()
=======
    # ------------------------------------------------------------
    # OpenGL 初期化
    # ------------------------------------------------------------
    def initializeGL(self):
        self.ctx = moderngl.create_context()
>>>>>>> 941fe4942b97dcdad64f5aa145809e1d66a430b8

    def update_frame(self):
        raw = np.array(self.sct.grab(self.mon))
        if raw is None or raw.size == 0:
            return

        frame_cpu = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2RGB)

        total_w, total_h = self.virtual_size
        geom_tgt = self.target_screen.geometry()
        part_w, part_h = geom_tgt.width(), geom_tgt.height()

        # === キャプチャ範囲：自分の担当 + 10% 重複 ===
        blend_ratio = 0.10
        overlap_px = int(part_w * blend_ratio)
        x_start = int((self.offset_x / total_w) * frame_cpu.shape[1]) - overlap_px
        x_end = int(((self.offset_x + part_w) / total_w) * frame_cpu.shape[1]) + overlap_px

        x_start = max(0, x_start)
        x_end = min(frame_cpu.shape[1], x_end)
        sub_cpu = frame_cpu[:, x_start:x_end]

        # === GPU resize ====================================
        if self.use_gpu:
            try:
                # 正しい GPU パス：GpuMat を使って upload → cv2.cuda.resize → download
                gsrc = cv2.cuda_GpuMat()
                gsrc.upload(sub_cpu)
                gresized = cv2.cuda.resize(gsrc, (part_w, part_h))
                resized = gresized.download()
            except Exception as e:
                log(f"[WARN] GPU resize failed, fallback to CPU resize: {e}")
                resized = cv2.resize(sub_cpu, (part_w, part_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = cv2.resize(sub_cpu, (part_w, part_h), interpolation=cv2.INTER_LINEAR)
        # ===================================================

<<<<<<< HEAD
        # === 歪み補正（warp_map は CPUのまま） ============
        warped = warp_image(resized, warp_info=self.warp_info)
        if warped is None:
            return

        # === フェード（CPU） ==============================
        if self.fade_enabled:
            h, w = warped.shape[:2]
            fade = np.ones((h, w), dtype=np.float32)
            blend_w = int(w * 0.10)
=======
        # --- 編集画面全体テクスチャ
        sg = self.source_screen.geometry()
        self.video_tex = self.ctx.texture(
            (sg.width(), sg.height()), 4
        )
        self.video_tex.swizzle = "BGRA"
        self.video_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # --- warp map（常に slice サイズ基準）
        map_x, map_y = self.warp_info
        uv = np.dstack([
            map_x / float(self.slice_size[0]),
            map_y / float(self.slice_size[1])
        ]).astype("f4")
>>>>>>> 941fe4942b97dcdad64f5aa145809e1d66a430b8

            for x in range(blend_w):
                alpha = x / float(blend_w)
                fade[:, x] *= alpha
                fade[:, -x - 1] *= alpha

            warped = (warped.astype(np.float32) * fade[..., None]).astype(np.uint8)
        # =================================================

<<<<<<< HEAD
        # === 出力 ========================================
        h, w, ch = warped.shape
        bytes_per_line = ch * w
        qt_image = QImage(warped.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qt_image))
=======
    # ------------------------------------------------------------
    # 描画
    # ------------------------------------------------------------
    def paintGL(self):
        img = self.sct.grab(self.monitor)
        self.video_tex.write(img.raw)
        self.video_tex.use(0)
        self.warp_tex.use(1)
        self.vao.render(moderngl.TRIANGLE_STRIP)
>>>>>>> 941fe4942b97dcdad64f5aa145809e1d66a430b8


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
<<<<<<< HEAD
    parser.add_argument("--mode", choices=["perspective", "warp_map"], default="perspective")
    parser.add_argument("--blend", action="store_true", help="Enable alpha blending")
=======
    parser.add_argument("--mode", default="warp_map")
    parser.add_argument("--blend", action="store_true")  # 将来拡張用
>>>>>>> 941fe4942b97dcdad64f5aa145809e1d66a430b8
    args = parser.parse_args()

    app = QApplication(sys.argv)

    # --- 入力された source / targets を内部IDに統一 ---
    src_vid = get_virtual_id(args.source)
    tgt_vids = [get_virtual_id(t) for t in args.targets]

<<<<<<< HEAD
    if not src_vid:
        print(f"❌ ソース {args.source} の内部ID変換に失敗")
        sys.exit(1)

    args.source = src_vid
    args.targets = [t for t in tgt_vids if t]


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
=======
    src_geo = source.geometry()
    num_targets = len(args.targets)

    slice_w = src_geo.width() // num_targets
    slice_h = src_geo.height()
>>>>>>> 941fe4942b97dcdad64f5aa145809e1d66a430b8

    windows = []
    offset_x = 0

<<<<<<< HEAD
    for name in args.targets:
        if name not in screens_by_name:
            print(f"⚠️ ターゲットディスプレイが見つかりません: {name}")
            continue

        target_screen = screens_by_name[name]
        fade_enabled = args.blend and len(args.targets) > 1

        warp_info = prepare_warp(name, args.mode,
                                 (target_screen.geometry().width(), target_screen.geometry().height()),
                                 load_points_func=load_points, log_func=log)

        if warp_info is None:
            print(f"⚠️ {name} の warp 情報がありません。スキップします。")
            continue

        print(f"🎥 {args.source} → {name} 出力 (fade={fade_enabled})")

        window = DisplayWindow(
            source_screen, target_screen, args.mode,
            offset_x, virtual_size,
            warp_info_all=warp_info,
            fade_enabled=fade_enabled
=======
    for i, t in enumerate(args.targets):
        scr = screens[get_virtual_id(t)]

        offset_x = i * slice_w  # ★ 将来用（warp では使用しない）

        map_x, map_y = prepare_warp(
            t,
            args.mode,
            src_size=(slice_w, slice_h),
            # src_offset_x=offset_x,  # ← 設計変更により不使用
            load_points_func=load_points,
            log_func=log
>>>>>>> 941fe4942b97dcdad64f5aa145809e1d66a430b8
        )
        windows.append(window)
        offset_x += target_screen.geometry().width()

<<<<<<< HEAD
    if not windows:
        print("❌ 出力ディスプレイがありません。終了します。")
        sys.exit(1)
=======
        win = GLDisplayWindow(
            source,
            scr,
            offset_x,
            (slice_w, slice_h),
            (map_x, map_y)
        )
        win.show()
        windows.append(win)
>>>>>>> 941fe4942b97dcdad64f5aa145809e1d66a430b8

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
