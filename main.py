import sys
import os
import re
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
    QFileDialog, QDialog, QCheckBox,
    QDialogButtonBox, QMessageBox, QComboBox
)
import pygetwindow as gw
from grid_editor import GridEditorDialog

SETTINGS_DIR = "C:/Users/vrlab/.vscode/nukunuku/Sotsuken/settings"  # 保存先ディレクトリ

# Windowsのファイル名で使用できない文字をアンダースコアに置き換える
def sanitize_filename(name):
    return re.sub(r'[\\/:*?"<>|]', '_', name)

# ディスプレイ選択ダイアログ
class DisplaySelectionDialog(QDialog):
    # ディスプレイ選択ダイアログの初期化
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
    
    # 選択されたディスプレイのリストを返す
    def selected_displays(self):
        return [cb.text() for cb in self.checkboxes if cb.isChecked()]

    # 実際のディスプレイを認識して選択できるようにする
    def select_displays(self):
        screens = QApplication.screens()
        display_names = [f"{i}: {screen.name()}" for i, screen in enumerate(screens)]

        dialog = DisplaySelectionDialog(display_names, self)
        if dialog.exec_():
            selected = dialog.selected_displays()
            if not selected:
                return

            for sel in selected:
                selected_index = int(sel.split(":")[0])
                selected_screen = screens[selected_index]
                geometry = selected_screen.geometry()
                display_name = selected_screen.name()
                config = self.parent().load_display_config(display_name) if self.parent() else None

                editor = GridEditorDialog(
                    display_name=display_name,
                    config=config,
                    screen_geometry=geometry
                )
                editor.move(geometry.topLeft())
                editor.showFullScreen()
                if editor.exec_():
                    if self.parent():
                        self.parent().save_display_config(display_name, editor.get_current_config())

# Mainウィンドウ
class MainWindow(QMainWindow):
    # Mainウィンドウの初期化
    def __init__(self):
        super().__init__()
        self.setWindowTitle("プロジェクター歪み補正アプリ")
        self.setGeometry(200, 200, 400, 300)

        layout = QVBoxLayout()

        self.display_button = QPushButton("ディスプレイ選択")
        self.display_button.clicked.connect(self.select_displays)

        self.media_button = QPushButton("メディア選択")
        self.media_button.clicked.connect(self.select_media)

        self.object_button = QPushButton("オブジェクト選択")
        self.object_button.clicked.connect(self.select_object)

        layout.addWidget(self.display_button)
        layout.addWidget(self.media_button)
        layout.addWidget(self.object_button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    # ディスプレイ選択ダイアログを表示し、選択されたディスプレイの設定を読み込み、編集ダイアログを表示する
    def select_displays(self):
        screens = QApplication.screens()
        displays = [f"{i}: {screen.name()}" for i, screen in enumerate(screens)]
        dialog = DisplaySelectionDialog(displays, self)
        if dialog.exec_():
            selected = dialog.selected_displays()

            for display in selected:
                # display名からインデックスを取得
                selected_index = int(display.split(":")[0])
                selected_screen = screens[selected_index]
                geometry = selected_screen.geometry()
                display_name = selected_screen.name()
                config = self.load_display_config(display_name)
                editor = GridEditorDialog(
                    display_name=display_name,
                    config=config,
                    screen_geometry=geometry
                )
                editor.move(geometry.topLeft())
                editor.showFullScreen()
                if editor.exec_():
                    self.save_display_config(display_name, editor.get_current_config())

    # 設定をJSONから読み込み、保存する
    def load_display_config(self, display_name):
        """設定をJSONから読み込む"""
        os.makedirs(SETTINGS_DIR, exist_ok=True)
        safe_name = sanitize_filename(display_name)
        path = os.path.join(SETTINGS_DIR, f"{safe_name}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def save_display_config(self, display_name, config):
        """設定をJSONに保存する"""
        os.makedirs(SETTINGS_DIR, exist_ok=True)
        safe_name = sanitize_filename(display_name)
        path = os.path.join(SETTINGS_DIR, f"{safe_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    # メディア選択ダイアログを表示し、選択されたメディアの種類に応じてファイル選択ダイアログを開く
    def select_media(self):
        options = ["画像", "動画", "ウィンドウ"]
        dialog = QDialog(self)
        dialog.setWindowTitle("メディアの種類を選択")
        layout = QVBoxLayout()

        combo = QComboBox()
        combo.addItems(options)
        layout.addWidget(combo)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)
        dialog.setLayout(layout)

        # ボタンのクリックイベントを設定
        def accept():
            choice = combo.currentText()
            dialog.accept()
            if choice == "画像":
                file, _ = QFileDialog.getOpenFileName(self, "画像ファイルを選択", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)")
                if file:
                    QMessageBox.information(self, "選択された画像", file)
            elif choice == "動画":
                file, _ = QFileDialog.getOpenFileName(self, "動画ファイルを選択", "", "Videos (*.mp4 *.avi *.mov *.mkv)")
                if file:
                    QMessageBox.information(self, "選択された動画", file)
            elif choice == "ウィンドウ":
                windows = gw.getAllTitles()
                msg = "\n".join([w for w in windows if w.strip()])
                QMessageBox.information(self, "起動中ウィンドウ", msg or "検出できませんでした")

        buttons.accepted.connect(accept)
        buttons.rejected.connect(dialog.reject)
        dialog.exec_()

    # 3Dオブジェクト選択ダイアログを表示し、選択されたファイルのパスを表示する
    def select_object(self):
        file, _ = QFileDialog.getOpenFileName(self, "3Dオブジェクトを選択", "", "3D Files (*.stl *.obj *.ply *.glb *.gltf)")
        if file:
            QMessageBox.information(self, "選択されたファイル", file)

# PyQt5アプリケーションのエントリーポイント
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
