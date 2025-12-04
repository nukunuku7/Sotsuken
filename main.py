# main.py（blend自動判定付き）
# ==============================
# 仮想環境有効かコマンド
# sovenv\Scripts\activate
# 
# 使用するプロジェクトを変える際は、仮想環境を無効化してから行うこと
# deactivate
# ==============================

import sys
import os
import json
import subprocess

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
    QLabel, QMessageBox, QComboBox, QListWidget, QListWidgetItem
)
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtCore import Qt

from editor.grid_utils import sanitize_filename, auto_generate_from_environment

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
SETTINGS_DIR = os.path.join(CONFIG_DIR, "projector_profiles")
TEMP_DIR = os.path.join(BASE_DIR, "temp")

os.makedirs(SETTINGS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

EDIT_PROFILE_PATH = os.path.join(CONFIG_DIR, "edit_profile.json")


def save_edit_profile(display_name):
    with open(EDIT_PROFILE_PATH, "w") as f:
        json.dump({"display": display_name}, f)


def load_edit_profile():
    if os.path.exists(EDIT_PROFILE_PATH):
        with open(EDIT_PROFILE_PATH, "r") as f:
            return json.load(f).get("display")
    return None


# --- ディスプレイの実機配置とPyQt名のマッピング ---
# --- 動的ディスプレイ検出関数 ---
def get_display_mapping():
    """
    接続されているスクリーンを x 座標順（左→右）に並べ、
    '1','2','3',... のラベルを割り当てた dict を返す。
    値は PyQt の screen.name()（例 '\\\\.\\DISPLAY6'）となる。
    """
    screens = QGuiApplication.screens()
    # sort by x coordinate (left -> right)
    sorted_screens = sorted(screens, key=lambda s: s.geometry().x())
    mapping = {}
    for idx, s in enumerate(sorted_screens, start=1):
        mapping[str(idx)] = s.name()
    return mapping


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
        """Windows番号順でリストを作成（実機の左→右を 1..N に割当）"""
        self.projector_list.clear()
        display_map = get_display_mapping()  # 動的マッピングを取得
        # store for later use (launch 等で同じ割当を使うため)
        self.display_map = display_map

        for win_id, pyqt_name in display_map.items():
            # 編集用ディスプレイは除外
            if pyqt_name == self.edit_display_name:
                continue
            item = QListWidgetItem(f"ディスプレイ{win_id} ({pyqt_name})")
            item.setData(Qt.UserRole, pyqt_name)  # 内部データとしてPyQt名を保持
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
            "各補正ディスプレイでグリッドを微調整し、\n"
            "下のボタンを押してからウィンドウを閉じてください。\n"
            "escキーでグリッド画面が終了します。"
        )

        self.instruction_window = QWidget()
        self.instruction_window.setWindowTitle("保存操作ガイド")
        self.instruction_window.setGeometry(geom.x() + 100, geom.y() + 100, 520, 240)

        layout = QVBoxLayout()
        label = QLabel(f"補正モード：{mode}\n{msg}")
        label.setStyleSheet("font-size: 16px; padding: 20px; background-color: #222; color: white;")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        save_button = QPushButton("チェック済みディスプレイを保存")
        save_button.setStyleSheet("font-size: 14px; background-color: #00cc66; color: white; padding: 10px;")
        save_button.clicked.connect(lambda: self.force_save_grids(mode))
        layout.addWidget(save_button)

        self.instruction_window.setLayout(layout)
        self.instruction_window.show()

    def launch_editors(self):
        mode = self.mode_selector.currentText()
        auto_generate_from_environment(mode=mode)
        self.launch_instruction_window(mode)

        # ✅ チェックされたディスプレイだけ起動
        for i in range(self.projector_list.count()):
            item = self.projector_list.item(i)
            if item.checkState() == Qt.Checked:
                pyqt_name = item.data(Qt.UserRole)
                geom = None
                for screen in QGuiApplication.screens():
                    if screen.name() == pyqt_name:
                        geom = screen.geometry()
                        break
                if geom is None:
                    continue

                script_path = os.path.join(BASE_DIR, "editor",
                    "grid_editor_perspective.py" if mode == "perspective" else "grid_editor_warpmap.py"
                )
                lock_path = os.path.join(TEMP_DIR, f"editor_active_{sanitize_filename(pyqt_name, mode)}.lock")
                with open(lock_path, "w") as f:
                    f.write("active")

                cmd = [
                    sys.executable, script_path,
                    "--display", pyqt_name,
                    "--x", str(geom.x()), "--y", str(geom.y()),
                    "--w", str(geom.width()), "--h", str(geom.height())
                ]
                subprocess.Popen(cmd)

    def force_save_grids(self, mode):
        selected_names = []
        for i in range(self.projector_list.count()):
            item = self.projector_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_names.append(item.data(Qt.UserRole))

        if not selected_names:
            QMessageBox.warning(self, "警告", "保存対象のディスプレイが選択されていません")
            return

        from editor.grid_utils import auto_generate_from_environment
        auto_generate_from_environment(mode=mode, displays=selected_names)

        for name in selected_names:
            lock_path = os.path.join(TEMP_DIR, f"editor_active_{sanitize_filename(name, mode)}.lock")
            if os.path.exists(lock_path):
                os.remove(lock_path)

        QMessageBox.information(
            self, "保存完了",
            f"モード '{mode}' のグリッドを {', '.join(selected_names)} に保存しました。"
        )

    def launch_correction_display(self):
        selected_names = []
        for i in range(self.projector_list.count()):
            item = self.projector_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_names.append(item.data(Qt.UserRole))

        if not selected_names:
            QMessageBox.warning(self, "警告", "出力先ディスプレイが選択されていません")
            return

        mode = self.mode_selector.currentText()
        source_display = self.edit_display_name

        cmd = [
            sys.executable,
            os.path.join(BASE_DIR, "media_player_multi.py"),
            "--source", source_display,
            "--targets", *selected_names,
            "--mode", mode,
        ]

        # ✅ 複数ディスプレイならブレンド有効化
        if len(selected_names) > 1:
            cmd.append("--blend")

        subprocess.Popen(cmd)


# --- GPU 利用可否チェック（main 側の確認用） ---
def is_gpu_available_main():
    try:
        import cupy as cp
        # device count check
        cnt = cp.cuda.runtime.getDeviceCount()
        if cnt <= 0:
            return False
        # quick sanity op
        _ = cp.array([1], dtype=cp.int32) * 2
        return True
    except Exception:
        return False


# --- アプリ起動 ---
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # --- 起動時デバッグ出力 ---
    print("=== Display Mapping ===")
    screens = QGuiApplication.screens()
    for i, s in enumerate(screens):
        g = s.geometry()
        print(f"[{i}] {s.name()} : {g.width()}x{g.height()} at ({g.x()},{g.y()})")
    print("========================")

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
