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
    def __init__(self, screen, mode):
        super().__init__()
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_DeleteOnClose)

        self.display_name = screen.name()
        self.mode = mode

        # ディスプレイの位置とサイズ
        geom = screen.geometry()
        self.setGeometry(geom)

        self.label = QLabel(self)
        self.label.setGeometry(0, 0, geom.width(), geom.height())

        # MSSによるスクリーンキャプチャ設定
        self.sct = mss.mss()
        self.mon = {
            "left": geom.x(),
            "top": geom.y(),
            "width": geom.width(),
            "height": geom.height()
        }

        # warp情報を初期化（事前に保存されたグリッドを使用）
        self.warp_info = prepare_warp(
            self.display_name,
            self.mode,
            (geom.height(), geom.width())
        )

        # 30fps程度で更新
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # 約30fps

        self.showFullScreen()

    def update_frame(self):
        # ディスプレイの内容をキャプチャ
        frame = np.array(self.sct.grab(self.mon))[:, :, :3]  # BGRA→BGR

        # warp補正を適用
        warped = warp_image(frame, warp_info=self.warp_info)

        # PyQtに描画
        h, w, ch = warped.shape
        bytes_per_line = ch * w
        qt_image = QImage(
            warped.tobytes(),  # ← 修正: .data ではなく .tobytes()
            w, h, bytes_per_line,
            QImage.Format_RGB888
        ).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(qt_image))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--displays", nargs='+', required=True,
                        help="出力対象のディスプレイ名")
    parser.add_argument("--mode", choices=["perspective", "warp_map"],
                        default="perspective")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    screens = QGuiApplication.screens()
    windows = []

    for screen in screens:
        if screen.name() in args.displays:
            print(f"🎥 {screen.name()} をキャプチャして補正出力します (モード: {args.mode})")
            window = DisplayWindow(screen, mode=args.mode)
            windows.append(window)

    if not windows:
        print("❌ 指定されたディスプレイが見つかりません")
        sys.exit(1)

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
