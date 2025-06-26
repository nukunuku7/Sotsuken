# main.py（編集用ディスプレイ認識＋モード別初期グリッド生成対応）
import sys
import os
import re
import json
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
    QLabel, QMessageBox, QComboBox
)
from PyQt5.QtGui import QGuiApplication
from grid_utils import sanitize_filename

SETTINGS_DIR = "C:/Users/vrlab/.vscode/nukunuku/Sotsuken/settings"
EDIT_PROFILE_PATH = os.path.join(SETTINGS_DIR, "edit_profile.json")
os.makedirs(SETTINGS_DIR, exist_ok=True)

def sanitize_filename(name):
    return re.sub(r'[\\/:*?"<>|]', '_', name)

def save_edit_profile(display_name):
    with open(EDIT_PROFILE_PATH, "w") as f:
        json.dump({"display": display_name}, f)

def load_edit_profile():
    if os.path.exists(EDIT_PROFILE_PATH):
        with open(EDIT_PROFILE_PATH, "r") as f:
            return json.load(f).get("display")
    return None

def launch_grid_editor(display_name, geometry, mode):
    x, y, w, h = geometry
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "grid_editor.py"),
        "--mode", mode,
        "--display", display_name,
        "--x", str(x), "--y", str(y),
        "--w", str(w), "--h", str(h)
    ]
    subprocess.Popen(cmd)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("プロジェクター歪み補正アプリ")
        self.setGeometry(200, 200, 420, 300)

        layout = QVBoxLayout()

        self.label = QLabel("編集用ディスプレイ：未認識")
        layout.addWidget(self.label)

        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["perspective", "warp_map"])
        layout.addWidget(QLabel("補正方式を選択："))
        layout.addWidget(self.mode_selector)

        self.auto_edit_button = QPushButton("プロジェクター全台グリッド編集")
        self.auto_edit_button.clicked.connect(self.auto_launch_editors)
        layout.addWidget(self.auto_edit_button)

        self.launch_button = QPushButton("補正表示起動（出力反映）")
        self.launch_button.clicked.connect(self.launch_correction_display)
        layout.addWidget(self.launch_button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.init_display_info()

    def init_display_info(self):
        screen = QGuiApplication.primaryScreen()
        self.edit_display_name = screen.name()
        save_edit_profile(self.edit_display_name)
        self.label.setText(f"編集用ディスプレイ：{self.edit_display_name}")

    def auto_launch_editors(self):
        screens = QGuiApplication.screens()
        mode = self.mode_selector.currentText()
        for screen in screens:
            display_name = screen.name()
            if display_name == self.edit_display_name:
                continue
            geometry = screen.geometry()
            sanitized_name = sanitize_filename(display_name)  # 追加
            launch_grid_editor(sanitized_name, (geometry.x(), geometry.y(), geometry.width(), geometry.height()), mode)

    def launch_correction_display(self):
        screens = QGuiApplication.screens()
        secondary_screens = [s for s in screens if s.name() != self.edit_display_name]
        display_names = [s.name() for s in secondary_screens]

        if not display_names:
            QMessageBox.warning(self, "警告", "表示可能なプロジェクターが見つかりません")
            return

        mode = self.mode_selector.currentText()

        sanitized_names = list(map(sanitize_filename, display_names))  # 追加

        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "media_player_multi.py"),
            "--displays", *sanitized_names,  # 修正済み
            "--mode", mode
        ]
        subprocess.Popen(cmd)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
