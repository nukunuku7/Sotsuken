import sys
import argparse
import cv2
import numpy as np
import mss
from PyQt5.QtWidgets import QApplication, QLabel, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QGuiApplication
from warp_engine import warp_image, prepare_warp

class DisplayWindow(QWidget):
    def __init__(self, source_screen, target_screen, mode):
        super().__init__()
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_DeleteOnClose)

        self.source_name = source_screen.name()
        self.target_name = target_screen.name()
        self.mode = mode

        # 出力側ディスプレイのジオメトリ設定
        geom_out = target_screen.geometry()
        self.setGeometry(geom_out)
        self.label = QLabel(self)
        self.label.setGeometry(0, 0, geom_out.width(), geom_out.height())

        # キャプチャはメイン画面（または source_screen）
        self.sct = mss.mss()
        self.mon = {
            "left": source_screen.geometry().x(),
            "top": source_screen.geometry().y(),
            "width": source_screen.geometry().width(),
            "height": source_screen.geometry().height()
        }

        # warp情報を初期化
        self.warp_info = prepare_warp(
            self.target_name,
            self.mode,
            (geom_out.width(), geom_out.height())
        )

        # 30fps程度で更新
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)

        self.showFullScreen()

    def update_frame(self):
        # キャプチャ対象のディスプレイを取得
        raw = np.array(self.sct.grab(self.mon))
        if raw is None or raw.size == 0:
            return

        frame = raw[:, :, :3]  # BGRA -> BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # warp補正
        warped = warp_image(frame, warp_info=self.warp_info)
        if warped is None:
            return

        h, w, ch = warped.shape
        bytes_per_line = ch * w
        data = warped.tobytes()
        qt_image = QImage(data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qt_image))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="キャプチャ元ディスプレイ名（通常メイン）")
    parser.add_argument("--targets", nargs='+', required=True, help="補正出力先ディスプレイ名")
    parser.add_argument("--mode", choices=["perspective", "warp_map"], default="perspective")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    screens = QGuiApplication.screens()
    screen_dict = {s.name(): s for s in screens}

    if args.source not in screen_dict:
        print(f"❌ キャプチャ元ディスプレイが見つかりません: {args.source}")
        sys.exit(1)

    windows = []
    for t in args.targets:
        if t in screen_dict:
            print(f"🎥 {args.source} → {t} に補正出力 ({args.mode})")
            window = DisplayWindow(screen_dict[args.source], screen_dict[t], mode=args.mode)
            windows.append(window)
        else:
            print(f"⚠️ 出力ディスプレイが見つかりません: {t}")

    if not windows:
        sys.exit(1)

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
