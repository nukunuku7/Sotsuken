# main.py

import sys
import os
import json
import subprocess
import re

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
    QLabel, QMessageBox, QComboBox, QListWidget, QListWidgetItem, QCheckBox
)
from PyQt5.QtGui import QGuiApplication

from grid_utils import sanitize_filename

SETTINGS_DIR = "settings"
EDIT_PROFILE_PATH = os.path.join(SETTINGS_DIR, "edit_profile.json")
os.makedirs(SETTINGS_DIR, exist_ok=True)

def save_edit_profile(display_name):
    with open(EDIT_PROFILE_PATH, "w") as f:
        json.dump({"display": display_name}, f)

def load_edit_profile():
    if os.path.exists(EDIT_PROFILE_PATH):
        with open(EDIT_PROFILE_PATH, "r") as f:
            return json.load(f).get("display")
    return None

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("360°歪み補正プロジェクションシステム")
        self.setGeometry(200, 200, 480, 400)

        layout = QVBoxLayout()

        self.label = QLabel("編集用ディスプレイ：未認識")
        layout.addWidget(self.label)

        # 補正方式選択（射影変換 or 自由変形）
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["perspective", "warp_map"])
        layout.addWidget(QLabel("補正方式を選択："))
        layout.addWidget(self.mode_selector)

        # プロジェクター選択リスト
        self.projector_list = QListWidget()
        layout.addWidget(QLabel("補正出力先ディスプレイを選択："))
        layout.addWidget(self.projector_list)

        # 編集ボタン
        self.edit_button = QPushButton("グリッドエディター起動")
        self.edit_button.clicked.connect(self.launch_editors)
        layout.addWidget(self.edit_button)

        # 表示反映ボタン
        self.launch_button = QPushButton("補正表示 起動（出力反映）")
        self.launch_button.clicked.connect(self.launch_correction_display)
        layout.addWidget(self.launch_button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.edit_display_name = ""
        self.init_display_info()
        self.init_projector_list()

    def init_display_info(self):
        screen = QGuiApplication.primaryScreen()
        self.edit_display_name = screen.name()
        self.label.setText(f"編集用ディスプレイ：{self.edit_display_name}")
        save_edit_profile(self.edit_display_name)

    def init_projector_list(self):
        screens = QGuiApplication.screens()
        for screen in screens:
            if screen.name() == self.edit_display_name:
                continue
            item = QListWidgetItem(screen.name())
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.projector_list.addItem(item)

    def launch_editors(self):
        mode = self.mode_selector.currentText()
        script = "grid_editor_perspective.py" if mode == "perspective" else "grid_editor_warpmap.py"

        screens = QGuiApplication.screens()
        for screen in screens:
            if screen.name() == self.edit_display_name:
                continue
            geom = screen.geometry()
            cmd = [
                sys.executable,
                os.path.join(os.path.dirname(__file__), script),
                "--mode", mode,
                "--display", screen.name(),
                "--x", str(geom.x()), "--y", str(geom.y()),
                "--w", str(geom.width()), "--h", str(geom.height())
            ]
            subprocess.Popen(cmd)

    def launch_correction_display(self):
        selected_names = []
        for i in range(self.projector_list.count()):
            item = self.projector_list.item(i)
            if item.checkState():
                selected_names.append(item.text())

        if not selected_names:
            QMessageBox.warning(self, "警告", "出力先ディスプレイが選択されていません")
            return

        mode = self.mode_selector.currentText()
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "media_player_multi.py"),
            "--displays", *selected_names,
            "--mode", mode
        ]
        subprocess.Popen(cmd)

if __name__ == "__main__":
    from PyQt5.QtCore import Qt

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
