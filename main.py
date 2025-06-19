# main.py（編集ディスプレイ記憶 + ミニプレビュー機能付き）
import sys
import os
import re
import json
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
    QMessageBox, QDialog, QDialogButtonBox, QComboBox, QLabel
)
from PyQt5.QtGui import QGuiApplication

SETTINGS_DIR = "C:/Users/vrlab/.vscode/nukunuku/Sotsuken/settings"
PROFILE_PATH = os.path.join(SETTINGS_DIR, "edit_profile.json")


def sanitize_filename(name):
    return re.sub(r'[\\/:*?"<>|]', '_', name)


def launch_grid_editor(display_name, geometry):
    x, y, w, h = geometry
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "grid_editor.py"),
        "--mode", "editor",
        "--display", display_name,
        "--x", str(x), "--y", str(y),
        "--w", str(w), "--h", str(h)
    ]
    subprocess.Popen(cmd)


def get_config_path(display_name):
    return os.path.join(SETTINGS_DIR, f"{sanitize_filename(display_name)}.json")


def save_edit_screen_name(name):
    os.makedirs(SETTINGS_DIR, exist_ok=True)
    with open(PROFILE_PATH, 'w') as f:
        json.dump({"edit_screen": name}, f)


def load_edit_screen_name():
    if os.path.exists(PROFILE_PATH):
        with open(PROFILE_PATH, 'r') as f:
            data = json.load(f)
            return data.get("edit_screen")
    return None


class ScreenSelectionDialog(QDialog):
    def __init__(self, screens, parent=None):
        super().__init__(parent)
        self.setWindowTitle("編集用ディスプレイの選択")
        self.layout = QVBoxLayout()

        self.combo = QComboBox()
        self.screen_map = {}
        for i, screen in enumerate(screens):
            label = f"{i}: {screen.name()}"
            self.combo.addItem(label)
            self.screen_map[label] = screen

        self.layout.addWidget(self.combo)
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)
        self.setLayout(self.layout)

    def selected_screen(self):
        label = self.combo.currentText()
        return self.screen_map.get(label)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("プロジェクター歪み補正アプリ")
        self.setGeometry(200, 200, 400, 300)

        layout = QVBoxLayout()

        self.edit_button = QPushButton("編集用ディスプレイを選択")
        self.edit_button.clicked.connect(self.select_edit_screen)
        layout.addWidget(self.edit_button)

        self.auto_edit_button = QPushButton("プロジェクター全台グリッド編集")
        self.auto_edit_button.clicked.connect(self.auto_launch_editors)
        layout.addWidget(self.auto_edit_button)

        self.launch_button = QPushButton("補正表示起動（全画面キャプチャ）")
        self.launch_button.clicked.connect(self.launch_correction_display)
        layout.addWidget(self.launch_button)

        self.preview_label = QLabel("[ミニプレビュー機能：補正結果はプロジェクターに出力されます]")
        layout.addWidget(self.preview_label)

        self.edit_screen = None
        self.restore_edit_screen()

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def restore_edit_screen(self):
        saved_name = load_edit_screen_name()
        if not saved_name:
            return
        for screen in QGuiApplication.screens():
            if screen.name() == saved_name:
                self.edit_screen = screen
                self.preview_label.setText(f"編集ディスプレイ（復元）: {saved_name}")
                break

    def select_edit_screen(self):
        screens = QGuiApplication.screens()
        dialog = ScreenSelectionDialog(screens, self)
        if dialog.exec_():
            self.edit_screen = dialog.selected_screen()
            save_edit_screen_name(self.edit_screen.name())
            QMessageBox.information(self, "選択完了", f"編集用ディスプレイ: {self.edit_screen.name()}")
            self.preview_label.setText(f"編集ディスプレイ: {self.edit_screen.name()}")

    def auto_launch_editors(self):
        screens = QGuiApplication.screens()
        if not self.edit_screen:
            QMessageBox.warning(self, "警告", "編集用ディスプレイを先に選択してください")
            return
        for screen in screens:
            if screen.name() != self.edit_screen.name():
                geometry = screen.geometry()
                display_name = screen.name()
                os.makedirs(SETTINGS_DIR, exist_ok=True)
                json_path = get_config_path(display_name)
                if not os.path.exists(json_path):
                    with open(json_path, 'w') as f:
                        json.dump({}, f)
                launch_grid_editor(display_name, (geometry.x(), geometry.y(), geometry.width(), geometry.height()))

    def launch_correction_display(self):
        screens = QGuiApplication.screens()
        if not self.edit_screen:
            QMessageBox.warning(self, "警告", "編集用ディスプレイを先に選択してください")
            return
        secondary_screens = [s for s in screens if s.name() != self.edit_screen.name()]
        display_names = [s.name() for s in secondary_screens]

        if not display_names:
            QMessageBox.warning(self, "警告", "表示可能なプロジェクターが見つかりません")
            return

        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "media_player_multi.py"),
            "--displays", *display_names
        ]
        subprocess.Popen(cmd)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())