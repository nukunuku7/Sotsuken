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

import sys
import subprocess

def launch_grid_editor_on_display(display_name, geometry):
    x, y, w, h = geometry
    cmd = [
        sys.executable,
        "grid_editor.py",  # 修正：統合先ファイル名
        "--mode", "editor",  # 必須：統合されたファイルでの動作モード指定
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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("プロジェクター歪み補正アプリ")
        self.setGeometry(200, 200, 400, 300)

        layout = QVBoxLayout()

        self.display_button = QPushButton("ディスプレイ選択")
        self.display_button.clicked.connect(self.select_displays)

        self.window_button = QPushButton("ウィンドウ選択")
        self.window_button.clicked.connect(self.select_window)

        layout.addWidget(self.display_button)
        layout.addWidget(self.window_button)

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
                selected_index = int(display.split(":")[0])
                selected_screen = screens[selected_index]
                geometry = selected_screen.geometry()
                display_name = selected_screen.name()

                json_path = get_config_path(display_name)
                os.makedirs(SETTINGS_DIR, exist_ok=True)
                if not os.path.exists(json_path):
                    with open(json_path, 'w') as f:
                        json.dump({}, f)

                launch_grid_editor_on_display(
                    display_name,
                    (geometry.x(), geometry.y(), geometry.width(), geometry.height())
                )

    def select_window(self):
        import subprocess
        from media_player_core import MediaPlayer

        windows = [w.strip() for w in gw.getAllTitles() if w.strip()]
        if not windows:
            QMessageBox.information(self, "ウィンドウ選択", "ウィンドウが見つかりません")
            return

        win_dialog = QDialog(self)
        win_dialog.setWindowTitle("ウィンドウ選択")
        layout = QVBoxLayout()

        list_widget = QListWidget()
        list_widget.setSelectionMode(QListWidget.MultiSelection)
        for title in windows:
            item = QListWidgetItem(title)
            list_widget.addItem(item)
        layout.addWidget(list_widget)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)
        win_dialog.setLayout(layout)

        def on_accept():
            selected_items = list_widget.selectedItems()
            titles = [item.text() for item in selected_items]
            win_dialog.accept()

            if len(titles) == 1:
                player = MediaPlayer()
                player.play_window_by_title(titles[0])
            elif len(titles) > 1:
                cmd = [
                    sys.executable,
                    os.path.join(os.path.dirname(__file__), "media_player_multi.py"),
                    "--titles", *titles
                ]
                subprocess.Popen(cmd)
            else:
                QMessageBox.warning(self, "選択エラー", "ウィンドウが選択されていません")

        buttons.accepted.connect(on_accept)
        buttons.rejected.connect(win_dialog.reject)
        win_dialog.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
