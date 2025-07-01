# main.py（修正済み：補助ウィンドウ表示付き）

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
from PyQt5.QtCore import Qt

from grid_utils import sanitize_filename, auto_generate_from_environment

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

        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["perspective", "warp_map"])
        layout.addWidget(QLabel("補正方式を選択："))
        layout.addWidget(self.mode_selector)

        self.projector_list = QListWidget()
        layout.addWidget(QLabel("補正出力先ディスプレイを選択："))
        layout.addWidget(self.projector_list)

        self.edit_button = QPushButton("グリッドエディター起動")
        self.edit_button.clicked.connect(self.launch_editors)
        layout.addWidget(self.edit_button)

        self.launch_button = QPushButton("補正表示 起動（出力反映）")
        self.launch_button.clicked.connect(self.launch_correction_display)
        layout.addWidget(self.launch_button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.edit_display_name = ""
        self.instruction_window = None
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

    def launch_instruction_window(self, mode):
        geom = None
        for screen in QGuiApplication.screens():
            if screen.name() == self.edit_display_name:
                geom = screen.geometry()
                break
        if geom is None:
            return

        msg = (
            "各補正ディスプレイでグリッドを微調整し、"
            "下のボタンを押してからウィンドウを閉じてください。"
        )

        self.instruction_window = QWidget()
        self.instruction_window.setWindowTitle("保存操作ガイド")
        self.instruction_window.setGeometry(geom.x() + 100, geom.y() + 100, 520, 240)

        layout = QVBoxLayout()

        label = QLabel(f"補正モード：{mode} {msg}")
        label.setStyleSheet("font-size: 16px; padding: 20px; background-color: #222; color: white;")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        save_button = QPushButton("すべてのグリッドを保存")
        save_button.setStyleSheet("font-size: 14px; background-color: #00cc66; color: white; padding: 10px;")
        save_button.clicked.connect(lambda: self.force_save_grids(mode))
        layout.addWidget(save_button)

        self.instruction_window.setLayout(layout)
        self.instruction_window.show()

    def launch_editors(self):
        mode = self.mode_selector.currentText()
        script = "grid_editor_perspective.py" if mode == "perspective" else "grid_editor_warpmap.py"

        auto_generate_from_environment(mode=mode)
        self.launch_instruction_window(mode)

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

    def force_save_grids(self, mode):
        from grid_utils import auto_generate_from_environment
        auto_generate_from_environment(mode=mode)
        QMessageBox.information(self, "保存完了", f"モード '{mode}' のグリッドを全ディスプレイに保存しました。")

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
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
