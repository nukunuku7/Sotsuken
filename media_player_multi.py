# media_player_multi.py

import sys
import argparse
import cv2
import numpy as np
import mss
from PyQt5.QtWidgets import QApplication, QLabel, QWidget
from PyQt5.QtGui import QImage, QPixmap, QGuiApplication
from PyQt5.QtCore import QTimer, Qt
from warp_engine import warp_image, prepare_warp


class DisplayWindow(QWidget):
    def __init__(self, source_screen, target_screen, mode, offset_x, virtual_size, fade_enabled=False):
        super().__init__()
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_DeleteOnClose)

        self.source_screen = source_screen
        self.target_screen = target_screen
        self.mode = mode
        self.offset_x = offset_x
        self.virtual_size = virtual_size
        self.fade_enabled = fade_enabled

        # 🎯 各ディスプレイ専用 warp 情報をロード
        self.warp_info = prepare_warp(target_screen.name(), mode)

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

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)
        self.showFullScreen()

    def update_frame(self):
        raw = np.array(self.sct.grab(self.mon))
        if raw is None or raw.size == 0:
            return

        frame = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2RGB)
        total_w, total_h = self.virtual_size
        geom_tgt = self.target_screen.geometry()
        part_w = geom_tgt.width()

        # 自分の担当領域を抽出
        x_start = int((self.offset_x / total_w) * frame.shape[1])
        x_end = int(((self.offset_x + part_w) / total_w) * frame.shape[1])
        sub_frame = frame[:, x_start:x_end]

        # 🌀 各ディスプレイ専用 warp 適用
        warped = warp_image(sub_frame, warp_info=self.warp_info)
        if warped is None:
            return

        # ブレンドありの場合フェードマスク適用
        if self.fade_enabled:
            h, w = warped.shape[:2]
            fade = np.ones((h, w), dtype=np.float32)
            blend_w = max(int(w * 0.1), 1)
            for x in range(blend_w):
                alpha = x / float(blend_w)
                fade[:, x] *= alpha
                fade[:, -x - 1] *= alpha
            warped = (warped.astype(np.float32) * fade[..., None]).astype(np.uint8)

        h, w, ch = warped.shape
        bytes_per_line = ch * w
        qt_image = QImage(warped.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qt_image))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--mode", choices=["perspective", "warp_map"], default="perspective")
    parser.add_argument("--blend", action="store_true", help="Enable edge blending between displays")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    screens = {s.name(): s for s in QGuiApplication.screens()}

    if args.source not in screens:
        print(f"❌ ソースディスプレイが見つかりません: {args.source}")
        sys.exit(1)

    source_screen = screens[args.source]

    # 仮想全体のサイズ（横連結）
    total_width = sum(screens[n].geometry().width() for n in args.targets if n in screens)
    max_height = max(screens[n].geometry().height() for n in args.targets if n in screens)
    virtual_size = (total_width, max_height)

    windows = []
    offset_x = 0

    for name in args.targets:
        if name not in screens:
            print(f"⚠️ ターゲットディスプレイが見つかりません: {name}")
            continue

        target_screen = screens[name]
        fade_enabled = args.blend and len(args.targets) > 1
        print(f"🎥 {args.source} → {name} 出力 (fade={fade_enabled})")

        # 各ディスプレイごとに独立してwarpをロード
        window = DisplayWindow(
            source_screen, target_screen, args.mode,
            offset_x, virtual_size,
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
