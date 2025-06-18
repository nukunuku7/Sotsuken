# media_player_multi.py（複数プロジェクター補正表示対応）
import sys
import numpy as np
import pygetwindow as gw
import win32gui
import win32con
import mss
import os
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from warp_engine import warp_image

SETTINGS_DIR = "C:/Users/vrlab/.vscode/nukunuku/Sotsuken/settings"

def activate_window(hwnd):
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    win32gui.SetForegroundWindow(hwnd)


def get_window_by_title(title_part):
    return next((w for w in gw.getWindowsWithTitle(title_part) if title_part in w.title), None)


class ProjectorWindow(QWidget):
    def __init__(self, window_title, screen_index=0, display_name="Display", window_id=None):
        super().__init__()
        self.setWindowTitle(f"Projector {window_id}")
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.window = get_window_by_title(window_title)
        if not self.window:
            raise RuntimeError(f"ウィンドウ '{window_title}' が見つかりません")

        activate_window(self.window._hWnd)
        self.sct = mss.mss()
        self.display_name = display_name

        screens = QApplication.screens()
        if screen_index >= len(screens):
            raise RuntimeError(f"指定されたディスプレイ index={screen_index} が存在しません")

        geometry = screens[screen_index].geometry()
        self.setGeometry(geometry)
        self.showFullScreen()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        img = self.capture_window()
        if img is None:
            return
        corrected = warp_image(img, self.display_name)
        self.display_image(corrected)

    def capture_window(self):
        hwnd = self.window._hWnd
        rect = win32gui.GetWindowRect(hwnd)
        x, y, x2, y2 = rect
        w, h = x2 - x, y2 - y
        monitor = {"top": y, "left": x, "width": w, "height": h}
        img = np.array(self.sct.grab(monitor))[:, :, :3]
        return img

    def display_image(self, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(qt_image))


def main(window_titles, display_names):
    app = QApplication(sys.argv)
    windows = []

    for i, title in enumerate(window_titles):
        try:
            name = display_names[i] if i < len(display_names) else f"Display{i}"
            win = ProjectorWindow(title, screen_index=i, display_name=name, window_id=i)
            windows.append(win)
        except Exception as e:
            print(f"[Error] {title}: {e}")

    if not windows:
        print("エラー: 有効なウィンドウがありません")
        sys.exit(1)

    sys.exit(app.exec_())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="複数プロジェクター用 歪み補正ビューア")
    parser.add_argument("--titles", nargs="+", required=True, help="キャプチャ対象のウィンドウタイトル（部分一致）")
    parser.add_argument("--displays", nargs="+", required=False, help="各ウィンドウに対応するディスプレイ名")
    args = parser.parse_args()

    display_names = args.displays if args.displays else [f"Display{i}" for i in range(len(args.titles))]
    main(args.titles, display_names)
