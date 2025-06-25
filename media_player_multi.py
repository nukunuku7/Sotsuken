# media_player_multi.py（編集ディスプレイ全体キャプチャ対応）
import sys
import numpy as np
import os
import json
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap, QGuiApplication
from PyQt5.QtCore import QTimer, Qt
from warp_engine import warp_image

SETTINGS_DIR = "C:/Users/vrlab/.vscode/nukunuku/Sotsuken/settings"
EDIT_PROFILE_PATH = os.path.join(SETTINGS_DIR, "edit_profile.json")

def load_edit_profile():
    if os.path.exists(EDIT_PROFILE_PATH):
        with open(EDIT_PROFILE_PATH, "r") as f:
            return json.load(f).get("display")
    return None

class ProjectorWindow(QWidget):
    def __init__(self, display_name, screen, edit_screen, window_id, mode="perspective"):
        super().__init__()
        self.setWindowTitle(f"Projector {window_id}")
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.display_name = display_name
        self.screen = screen
        self.edit_screen = edit_screen
        self.mode = mode

        self.geometry = screen.geometry()
        self.setGeometry(self.geometry)
        self.showFullScreen()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        img = self.capture_edit_screen()
        if img is None:
            return
        corrected = warp_image(img, self.display_name, mode=self.mode)
        self.display_image(corrected)

    def capture_edit_screen(self):
        if not self.edit_screen:
            return None
        pixmap = self.edit_screen.grabWindow(0)
        image = pixmap.toImage().convertToFormat(QImage.Format_RGB888)
        width = image.width()
        height = image.height()
        ptr = image.bits()
        ptr.setsize(height * width * 3)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
        return arr

    def display_image(self, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(qt_image))


def main(display_names, mode="perspective"):
    app = QApplication(sys.argv)
    screens = QGuiApplication.screens()
    edit_display_name = load_edit_profile()
    edit_screen = next((s for s in screens if s.name() == edit_display_name), None)

    windows = []
    for i, screen in enumerate(screens):
        name = screen.name()
        if name not in display_names or name == edit_display_name:
            continue
        try:
            win = ProjectorWindow(name, screen, edit_screen, window_id=i, mode=mode)
            windows.append(win)
        except Exception as e:
            print(f"[Error] {name}: {e}")

    if not windows:
        print("エラー: 有効なウィンドウがありません")
        sys.exit(1)

    sys.exit(app.exec_())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="複数プロジェクター用 歪み補正ビューア")
    parser.add_argument("--displays", nargs="+", required=True, help="補正出力対象ディスプレイ名")
    parser.add_argument("--mode", choices=["perspective", "warp_map"], default="perspective",
                        help="補正方法のモード")
    args = parser.parse_args()

    main(args.displays, mode=args.mode)
