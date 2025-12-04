import sys
import argparse
import cv2
import numpy as np
import mss
from PyQt5.QtWidgets import QApplication, QLabel, QWidget
from PyQt5.QtGui import QImage, QPixmap, QGuiApplication
from PyQt5.QtCore import QTimer, Qt

from editor.grid_utils import load_points, log
from warp_engine import warp_image, prepare_warp

# === GPU 自動検出 ==================================================
try:
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


class DisplayWindow(QWidget):
    def __init__(self, source_screen, target_screen, mode, offset_x, virtual_size,
                 warp_info_all=None, fade_enabled=False):
        super().__init__()
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_DeleteOnClose)

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
        }

        # warp 情報
        points_local = load_points(target_screen.name(), mode)
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
                display_name=target_screen.name(),
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

        self.showFullScreen()

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
            gpu_frame = cuda_GpuMat = cuda_GpuMat = cuda_GpuMat = cuda_GpuMat
            gpu_frame = cuda_GpuMat = cuda.GpuMat()
            gpu_frame.upload(sub_cpu)

            gpu_resized = cuda.resize(gpu_frame, (part_w, part_h))
            resized = gpu_resized.download()
        else:
            resized = cv2.resize(sub_cpu, (part_w, part_h), interpolation=cv2.INTER_LINEAR)
        # ===================================================

        # === 歪み補正（warp_map は CPUのまま） ============
        warped = warp_image(resized, warp_info=self.warp_info)
        if warped is None:
            return

        # === フェード（CPU） ==============================
        if self.fade_enabled:
            h, w = warped.shape[:2]
            fade = np.ones((h, w), dtype=np.float32)
            blend_w = int(w * 0.10)

            for x in range(blend_w):
                alpha = x / float(blend_w)
                fade[:, x] *= alpha
                fade[:, -x - 1] *= alpha

            warped = (warped.astype(np.float32) * fade[..., None]).astype(np.uint8)
        # =================================================

        # === 出力 ========================================
        h, w, ch = warped.shape
        bytes_per_line = ch * w
        qt_image = QImage(warped.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qt_image))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--mode", choices=["perspective", "warp_map"], default="perspective")
    parser.add_argument("--blend", action="store_true", help="Enable alpha blending")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    screens = {s.name(): s for s in QGuiApplication.screens()}

    if args.source not in screens:
        print(f"❌ ソースディスプレイが見つかりません: {args.source}")
        sys.exit(1)

    source_screen = screens[args.source]
    total_width = sum(screens[n].geometry().width() for n in args.targets if n in screens)
    max_height = max(screens[n].geometry().height() for n in screens if n in screens)
    virtual_size = (total_width, max_height)

    windows = []
    offset_x = 0

    for name in args.targets:
        if name not in screens:
            print(f"⚠️ ターゲットディスプレイが見つかりません: {name}")
            continue

        target_screen = screens[name]
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
        )
        windows.append(window)
        offset_x += target_screen.geometry().width()

    if not windows:
        print("❌ 出力ディスプレイがありません。終了します。")
        sys.exit(1)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
