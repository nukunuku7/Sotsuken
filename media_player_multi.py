# media_player_multi.py（PC全体を補正し各ディスプレイに分配表示）
import sys
import numpy as np
import mss
import os
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from warp_engine import warp_image

SETTINGS_DIR = "C:/Users/vrlab/.vscode/nukunuku/Sotsuken/settings"

class ProjectorWindow(QWidget):
    def __init__(self, display_index=0, display_name="Display", window_id=None):
        super().__init__()
        self.setWindowTitle(f"Projector {window_id}")
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.display_name = display_name
        self.display_index = display_index
        self.sct = mss.mss()

        screens = QApplication.screens()
        if display_index >= len(screens):
            raise RuntimeError(f"ディスプレイ index={display_index} が存在しません")

        self.geometry_rect = screens[display_index].geometry()
        self.setGeometry(self.geometry_rect)
        self.showFullScreen()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        full_img = self.capture_desktop()
        if full_img is None:
            return
        cropped = self.crop_to_screen(full_img)
        corrected = warp_image(cropped, self.display_name)
        self.display_image(corrected)

    def capture_desktop(self):
        monitor = self.sct.monitors[0]  # 全デスクトップ
        img = np.array(self.sct.grab(monitor))[:, :, :3]
        return img

    def crop_to_screen(self, full_img):
        gx, gy, gw, gh = self.geometry_rect.x(), self.geometry_rect.y(), self.geometry_rect.width(), self.geometry_rect.height()
        return full_img[gy:gy+gh, gx:gx+gw]

    def display_image(self, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(qt_image))


def main(display_names):
    app = QApplication(sys.argv)
    screens = QApplication.screens()
    windows = []

    for i, name in enumerate(display_names):
        try:
            win = ProjectorWindow(display_index=i, display_name=name, window_id=i)
            windows.append(win)
        except Exception as e:
            print(f"[Error] Display {i}: {e}")

    if not windows:
        print("エラー: 有効なウィンドウがありません")
        sys.exit(1)

    sys.exit(app.exec_())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="全画面補正ビューア（分配方式）")
    parser.add_argument("--displays", nargs="+", required=True, help="各補正対象のディスプレイ名")
    args = parser.parse_args()

    main(args.displays)
