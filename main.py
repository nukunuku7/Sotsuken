# main.py (final)
import sys
import os
import json
import subprocess
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
    QLabel, QMessageBox, QComboBox, QListWidget, QListWidgetItem
)
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtCore import Qt

from editor.grid_utils import (
    auto_generate_from_environment, get_virtual_id
)

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = BASE_DIR / "config"
SETTINGS_DIR = CONFIG_DIR / "projector_profiles"
TEMP_DIR = BASE_DIR / "temp"

SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)
EDIT_PROFILE_PATH = CONFIG_DIR / "edit_profile.json"
EDIT_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)

def save_edit_profile(display_name):
    with open(EDIT_PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump({"display": display_name}, f)

def load_edit_profile():
    if EDIT_PROFILE_PATH.exists():
        with open(EDIT_PROFILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("display")
    return None

def get_display_mapping():
    screens = QGuiApplication.screens()
    ordered = sorted(screens, key=lambda s: s.geometry().x())
    mapping = {}
    for idx, s in enumerate(ordered, start=1):
        mapping[str(idx)] = s.name()
    return mapping

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("360°歪み補正プロジェクションシステム")
        self.setGeometry(200, 200, 520, 420)

        layout = QVBoxLayout()
        self.label = QLabel("編集用ディスプレイ：未認識")
        layout.addWidget(self.label)

        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["perspective", "warp_map"])
        layout.addWidget(QLabel("補正方式を選択："))
        layout.addWidget(self.mode_selector)

        self.projector_list = QListWidget()
        layout.addWidget(QLabel("補正出力先ディスプレイを選択：(複数選択可)"))
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
        if screen:
            self.edit_display_name = screen.name()
            self.label.setText(f"編集用ディスプレイ：{self.edit_display_name}")
            save_edit_profile(self.edit_display_name)

    def init_projector_list(self):
        self.projector_list.clear()
        display_map = get_display_mapping()
        self.display_map = display_map

        for win_id, pyqt_name in display_map.items():
            if pyqt_name == self.edit_display_name:
                continue
            virt = get_virtual_id(pyqt_name)
            item = QListWidgetItem(f"ディスプレイ{win_id} ({pyqt_name})  [{virt}]")
            item.setData(Qt.UserRole, pyqt_name)
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

        # 選択されたディスプレイ名を取得
        selected_items = [
            self.projector_list.item(i)
            for i in range(self.projector_list.count())
            if self.projector_list.item(i).checkState() == Qt.Checked
        ]
        if not selected_items:
            QMessageBox.warning(self, "警告", "編集対象ディスプレイが選択されていません")
            return

        # ディスプレイ名とスクリーンX座標のリスト作成
        selected_displays = []
        for item in selected_items:
            pyqt_name = item.data(Qt.UserRole)
            geom = None
            for screen in QGuiApplication.screens():
                if screen.name() == pyqt_name:
                    geom = screen.geometry()
                    break
            if geom:
                selected_displays.append((pyqt_name, geom.x()))

        # 左→右順にソート
        selected_displays.sort(key=lambda t: t[1])

        # 選択ディスプレイの名前だけを抽出
        sorted_names = [d[0] for d in selected_displays]

        # まずJSONを生成（存在しない場合のみ）
        auto_generate_from_environment(mode=mode, displays=sorted_names)

        # 各ディスプレイの最初の点がマウスに引っ付くよう、順にセッション初期化
        for pyqt_name in sorted_names:
            virt = get_virtual_id(pyqt_name)
            total_points = 4 if mode == "perspective" else 36
            # grid_utils にあるセッション初期化関数（任意で作成済みなら呼び出す）
            try:
                from editor.grid_utils import init_editor_session
                init_editor_session(virt, total_points)
            except ImportError:
                pass  # 関数未定義でもエラーにしない

        # 説明ウィンドウ
        self.launch_instruction_window(mode)

        # 左→右順にエディター起動
        for pyqt_name, _ in selected_displays:
            geom = None
            for screen in QGuiApplication.screens():
                if screen.name() == pyqt_name:
                    geom = screen.geometry()
                    break
            if not geom:
                print(f"[WARN] 指定ディスプレイが見つかりません: {pyqt_name}")
                continue

            script = "grid_editor_perspective.py" if mode == "perspective" else "grid_editor_warpmap.py"
            script_path = str(BASE_DIR / "editor" / script)

            virt = get_virtual_id(pyqt_name)
            lock_path = TEMP_DIR / f"editor_active_{virt}_{mode}.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(lock_path, "w", encoding="utf-8") as f:
                    f.write("active")
            except Exception as e:
                print(f"[ERROR] ロックファイル作成失敗: {lock_path} ({e})")
                continue

            cmd = [
                sys.executable, script_path,
                "--display", pyqt_name,
                "--x", str(geom.x()), "--y", str(geom.y()),
                "--w", str(geom.width()), "--h", str(geom.height())
            ]
            subprocess.Popen(cmd)

            # 左→右順に少し待機して、次のディスプレイを起動
            import time
            time.sleep(0.5)  # 必要に応じて調整

    def force_save_grids(self, mode):
        selected_names = []
        for i in range(self.projector_list.count()):
            item = self.projector_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_names.append(item.data(Qt.UserRole))

        if not selected_names:
            QMessageBox.warning(self, "警告", "保存対象のディスプレイが選択されていません")
            return

        removed = []
        for name in selected_names:
            virt = get_virtual_id(name)
            lock_path = TEMP_DIR / f"editor_active_{virt}_{mode}.lock"
            if lock_path.exists():
                try:
                    lock_path.unlink()
                    removed.append(name)
                except Exception as e:
                    print(f"[WARN] ロックファイル削除に失敗しました: {lock_path} ({e})")

        if not removed:
            QMessageBox.information(self, "情報", "選択されたディスプレイに対するエディターのロックファイルが見つかりませんでした。")
            return

        QMessageBox.information(
            self, "保存命令送信",
            f"モード '{mode}' のグリッド保存命令を {', '.join(removed)} に送信しました。\nエディターが自動的に保存して終了します。"
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
        targets = [get_virtual_id(n) for n in selected_names]

        cmd = [
            sys.executable,
            str(BASE_DIR / "media_player_multi.py"),
            "--source", source_display,
            "--targets", *targets,
            "--mode", mode,
        ]
        if len(selected_names) > 1:
            cmd.append("--blend")
        subprocess.Popen(cmd)

def is_gpu_available_main():
    try:
        import cupy as cp
        cnt = cp.cuda.runtime.getDeviceCount()
        if cnt <= 0:
            return False
        _ = cp.array([1], dtype=cp.int32) * 2
        return True
    except Exception:
        return False

if __name__ == "__main__":
    app = QApplication(sys.argv)
    print("=== Display Mapping ===")
    screens = QGuiApplication.screens()
    for i, s in enumerate(screens):
        g = s.geometry()
        print(f"[{i}] {s.name()} : {g.width()}x{g.height()} at ({g.x()},{g.y()})")
    print("========================")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
