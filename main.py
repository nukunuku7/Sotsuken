# main.py（歪み補正表示機能付き）
import sys
import os
import re
import json
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
    QDialog, QCheckBox, QDialogButtonBox, QMessageBox, QListWidget,
    QListWidgetItem
)
import pygetwindow as gw

SETTINGS_DIR = "C:/Users/vrlab/.vscode/nukunuku/Sotsuken/settings"

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

class DisplaySelectionDialog(QDialog):
    def __init__(self, displays, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ディスプレイ選択")
        self.layout = QVBoxLayout()
        self.checkboxes = []
        for display in displays:
            cb = QCheckBox(display)
            self.layout.addWidget(cb)
            self.checkboxes.append(cb)
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)
        self.setLayout(self.layout)

    def selected_displays(self):
        return [cb.text() for cb in self.checkboxes if cb.isChecked()]

class WindowSelectionDialog(QDialog):
    def __init__(self, windows, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ウィンドウ選択")
        self.layout = QVBoxLayout()
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.MultiSelection)
        for title in windows:
            item = QListWidgetItem(title)
            self.list_widget.addItem(item)
        self.layout.addWidget(self.list_widget)
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)
        self.setLayout(self.layout)

    def selected_titles(self):
        return [item.text() for item in self.list_widget.selectedItems()]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("プロジェクター歪み補正アプリ")
        self.setGeometry(200, 200, 400, 300)

        layout = QVBoxLayout()

        self.display_button = QPushButton("ディスプレイ選択（編集モード）")
        self.display_button.clicked.connect(self.select_displays)
        layout.addWidget(self.display_button)

        self.launch_button = QPushButton("補正表示起動")
        self.launch_button.clicked.connect(self.launch_correction_display)
        layout.addWidget(self.launch_button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def select_displays(self):
        screens = QApplication.screens()
        displays = [f"{i}: {screen.name()}" for i, screen in enumerate(screens)]
        dialog = DisplaySelectionDialog(displays, self)
        if dialog.exec_():
            selected = dialog.selected_displays()
            for display in selected:
                idx = int(display.split(":")[0])
                screen = screens[idx]
                geometry = screen.geometry()
                display_name = screen.name()

                os.makedirs(SETTINGS_DIR, exist_ok=True)
                json_path = get_config_path(display_name)
                if not os.path.exists(json_path):
                    with open(json_path, 'w') as f:
                        json.dump({}, f)

                launch_grid_editor(display_name, (geometry.x(), geometry.y(), geometry.width(), geometry.height()))

    def launch_correction_display(self):
        windows = [w.strip() for w in gw.getAllTitles() if w.strip()]
        if not windows:
            QMessageBox.warning(self, "警告", "ウィンドウが見つかりません")
            return

        dialog = WindowSelectionDialog(windows, self)
        if dialog.exec_():
            selected_titles = dialog.selected_titles()
            screens = QApplication.screens()

            # 利用可能なディスプレイ名を順に取得
            available_display_names = [screen.name() for screen in screens]
            if len(selected_titles) > len(available_display_names):
                QMessageBox.warning(self, "警告", "選択されたウィンドウ数がディスプレイ数を超えています")
                return

            # ウィンドウ数に合わせてディスプレイ名をスライス
            display_names = available_display_names[:len(selected_titles)]

            if selected_titles:
                cmd = [
                    sys.executable,
                    os.path.join(os.path.dirname(__file__), "media_player_multi.py"),
                    "--titles", *selected_titles,
                    "--displays", *display_names
                ]
                subprocess.Popen(cmd)
            else:
                QMessageBox.warning(self, "警告", "ウィンドウが選択されていません")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())