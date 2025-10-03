# media_player_multi.py（修正済み：warp_info を初期化時に一度だけ作成）

import sys
import argparse
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QGuiApplication
from warp_engine import warp_image, prepare_warp

class DisplayWindow(QWidget):
    def __init__(self, screen, mode):
        super().__init__()
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_DeleteOnClose)

        self.display_name = screen.name()
        self.mode = mode

        geom = screen.geometry()
        self.setGeometry(geom)
        self.label = QLabel(self)
        self.label.setGeometry(0, 0, geom.width(), geom.height())

        self.blank_image = np.zeros((geom.height(), geom.width(), 3), dtype=np.uint8)
        self.warp_info = prepare_warp(self.display_name, self.mode, (geom.height(), geom.width()))

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.showFullScreen()

    def update_frame(self):
        img = self.blank_image.copy()
        cv2.circle(img, (img.shape[1] // 2, img.shape[0] // 2), 100, (255, 255, 255), -1)

        warped = warp_image(img.copy(), warp_info=self.warp_info)
        h, w, ch = warped.shape
        bytes_per_line = ch * w
        qt_image = QImage(warped.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(qt_image))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--displays", nargs='+', required=True, help="出力対象のディスプレイ名")
    parser.add_argument("--mode", choices=["perspective", "warp_map"], default="perspective")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    screens = QGuiApplication.screens()
    windows = []

    for screen in screens:
        if screen.name() in args.displays:
            window = DisplayWindow(screen, mode=args.mode)
            windows.append(window)

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
