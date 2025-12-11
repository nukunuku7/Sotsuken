# main.py（blend自動判定付き）
# ==============================
# 歪み補正システムのメインランチャーアプリケーション。
# エディターの起動、および補正済み映像の出力プログラム（media_player_multi.py）の起動を行います。
#
# 仮想環境有効かコマンド:
# sovenv\Scripts\activate
#
# 使用するプロジェクトを変える際は、仮想環境を無効化してから行うこと:
# deactivate
# ==============================

import sys
import os
import json
import subprocess # 外部プログラム (エディターやメディアプレイヤー) を実行するために使用

# PyQt5 GUIフレームワーク関連
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
    QLabel, QMessageBox, QComboBox, QListWidget, QListWidgetItem
)
from PyQt5.QtGui import QGuiApplication # ディスプレイ情報取得用
from PyQt5.QtCore import Qt # Qt.Checked や Qt.UserRole などの定数用

# グリッド設定ユーティリティ関数
from editor.grid_utils import sanitize_filename, auto_generate_from_environment, get_virtual_id

# --- パス定義と初期設定 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
SETTINGS_DIR = os.path.join(CONFIG_DIR, "projector_profiles") # グリッド設定ファイル（.json）の保存先
TEMP_DIR = os.path.join(BASE_DIR, "temp") # ロックファイルなど一時ファイルの保存先

# ディレクトリが存在しない場合は作成
os.makedirs(SETTINGS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

EDIT_PROFILE_PATH = os.path.join(CONFIG_DIR, "edit_profile.json") # 編集用ディスプレイ名を記録するファイル


# --- プロファイル保存/ロード関数 ---

def save_edit_profile(display_name):
    """メインウィンドウの編集用ディスプレイ名（プライマリスクリーン）を保存する。"""
    with open(EDIT_PROFILE_PATH, "w") as f:
        json.dump({"display": display_name}, f)


def load_edit_profile():
    """保存された編集用ディスプレイ名をロードする。"""
    if os.path.exists(EDIT_PROFILE_PATH):
        with open(EDIT_PROFILE_PATH, "r") as f:
            return json.load(f).get("display")
    return None


# --- ディスプレイ情報の動的検出 ---
def get_display_mapping():
    """
    QGuiApplicationから検出されたディスプレイをX座標順に並べ、
    数値ID (1, 2, 3...) と PyQt名 ('\\\\.\\DISPLAY1') のマッピングを返す。
    リスト表示のソートとデータ紐付けに使用。
    """
    screens = QGuiApplication.screens()
    sorted_screens = sorted(screens, key=lambda s: s.geometry().x()) # X座標が小さい順に並べる
    mapping = {}
    for idx, s in enumerate(sorted_screens, start=1):
        mapping[str(idx)] = s.name()
    return mapping


class MainWindow(QMainWindow):
    """
    システムのメインGUIウィンドウクラス。
    ディスプレイの選択、モードの選択、エディター/出力の起動ボタンを提供。
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("360°歪み補正プロジェクションシステム")
        self.setGeometry(200, 200, 480, 400) # ウィンドウの初期位置とサイズ

        # --- ウィジェットの初期化 ---
        layout = QVBoxLayout()
        self.label = QLabel("編集用ディスプレイ：未認識") # プライマリスクリーン表示
        layout.addWidget(self.label)

        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["perspective", "warp_map"]) # 補正モード選択
        layout.addWidget(QLabel("補正方式を選択："))
        layout.addWidget(self.mode_selector)

        self.projector_list = QListWidget() # 出力先ディスプレイのリスト
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

        self.edit_display_name = "" # 編集用ディスプレイのPyQt名
        self.instruction_window = None # エディター起動時のガイドウィンドウ

        # --- 初期処理 ---
        self.init_display_info()
        self.init_projector_list()

    def init_display_info(self):
        """プライマリスクリーンを編集用ディスプレイとして設定する。"""
        screen = QGuiApplication.primaryScreen()
        self.edit_display_name = screen.name()
        self.label.setText(f"編集用ディスプレイ：{self.edit_display_name}")
        save_edit_profile(self.edit_display_name) # プロファイルに保存

    def init_projector_list(self):
        """リストウィジェットに、編集用ディスプレイ以外の全ての検出ディスプレイを追加する。"""
        self.projector_list.clear()
        display_map = get_display_mapping()
        self.display_map = display_map

        for win_id, pyqt_name in display_map.items():
            # プライマリスクリーンは除外
            if pyqt_name == self.edit_display_name:
                continue
            
            # リストアイテムを作成し、チェックボックスを有効化
            item = QListWidgetItem(f"ディスプレイ{win_id} ({pyqt_name})")
            item.setData(Qt.UserRole, pyqt_name) # PyQt名（内部識別子）をデータとして保持
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.projector_list.addItem(item)

    def launch_instruction_window(self, mode):
        """グリッドエディター起動時に、操作ガイドウィンドウを表示する。"""
        # メインディスプレイの座標を取得し、その近くにガイドウィンドウを配置
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
        # ボタン押下でグリッド設定を強制保存し、ロックファイルを解除
        save_button.clicked.connect(lambda: self.force_save_grids(mode))
        layout.addWidget(save_button)

        self.instruction_window.setLayout(layout)
        self.instruction_window.show()

    def launch_editors(self):
        """選択されたディスプレイのグリッドエディターを起動する。"""
        mode = self.mode_selector.currentText()
        # 既存の設定がない場合、デフォルトのグリッド設定ファイルを自動生成/更新
        auto_generate_from_environment(mode=mode) 
        
        self.launch_instruction_window(mode)

        for i in range(self.projector_list.count()):
            item = self.projector_list.item(i)
            if item.checkState() == Qt.Checked:
                pyqt_name = item.data(Qt.UserRole)
                
                # ディスプレイの幾何学情報（座標とサイズ）を取得
                geom = None
                for screen in QGuiApplication.screens():
                    if screen.name() == pyqt_name:
                        geom = screen.geometry()
                        break
                if geom is None:
                    continue

                # 起動するエディタースクリプトをモードに応じて選択
                script_path = os.path.join(BASE_DIR, "editor",
                    "grid_editor_perspective.py" if mode == "perspective" else "grid_editor_warpmap.py"
                )

                # 仮想ID（D1, D2, D3...）を取得し、ロックファイル名を生成
                virt = get_virtual_id(pyqt_name)
                lock_path = os.path.join(TEMP_DIR, f"editor_active_{sanitize_filename(virt, mode)}.lock")
                
                # ロックファイルを作成し、エディターが起動中であることを示す
                with open(lock_path, "w") as f:
                    f.write("active")

                # エディターを別プロセスとして起動 (PyQt名、座標、サイズを引数で渡す)
                cmd = [
                    sys.executable, script_path,
                    "--display", pyqt_name,
                    "--x", str(geom.x()), "--y", str(geom.y()),
                    "--w", str(geom.width()), "--h", str(geom.height())
                ]
                subprocess.Popen(cmd) # 非同期で実行

    def force_save_grids(self, mode):
        """エディター終了前に、開いているグリッド設定を強制的に保存する。"""
        selected_names = []
        for i in range(self.projector_list.count()):
            item = self.projector_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_names.append(item.data(Qt.UserRole))

        if not selected_names:
            QMessageBox.warning(self, "警告", "保存対象のディスプレイが選択されていません")
            return

        # 再度、選択されたディスプレイに対してグリッド設定ファイルを更新・保存（上書き）
        auto_generate_from_environment(mode=mode, displays=selected_names)

        # 関連するロックファイルを削除し、エディターが終了したことを通知
        for name in selected_names:
            virt = get_virtual_id(name)
            lock_path = os.path.join(TEMP_DIR, f"editor_active_{sanitize_filename(virt, mode)}.lock")
            if os.path.exists(lock_path):
                os.remove(lock_path)

        QMessageBox.information(
            self, "保存完了",
            f"モード '{mode}' のグリッドを {', '.join(selected_names)} に保存しました。"
        )

    def launch_correction_display(self):
        """選択されたディスプレイに補正済み映像出力プログラムを起動する。"""
        selected_names = []
        for i in range(self.projector_list.count()):
            item = self.projector_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_names.append(item.data(Qt.UserRole))

        if not selected_names:
            QMessageBox.warning(self, "警告", "出力先ディスプレイが選択されていません")
            return

        mode = self.mode_selector.currentText()
        source_display = self.edit_display_name # 編集用ディスプレイをソースとする

        # ターゲットディスプレイのPyQt名を、プログラムが内部的に使う仮想ID（D2, D3, ...）に変換
        targets = [get_virtual_id(n) for n in selected_names]

        # media_player_multi.py を別プロセスで起動するためのコマンドを構築
        cmd = [
            sys.executable, # 現在の仮想環境のPython実行ファイルを使用
            os.path.join(BASE_DIR, "media_player_multi.py"),
            "--source", source_display, # ソースディスプレイ (PyQt名)
            "--targets", *targets, # ターゲットディスプレイリスト (仮想ID)
            "--mode", mode,
        ]

        # ターゲットが複数ある場合、ブレンド（フェード）を有効化するフラグを追加
        if len(selected_names) > 1:
            cmd.append("--blend")

        subprocess.Popen(cmd) # 非同期で実行


# --- GPU 利用可否チェック（main 側の確認用: 必要に応じて削除/コメントアウト可） ---
def is_gpu_available_main():
    """CuPyを使用してGPUが利用可能かを確認するユーティリティ関数。"""
    try:
        import cupy as cp
        cnt = cp.cuda.runtime.getDeviceCount()
        if cnt <= 0:
            return False
        # 実際に簡単な演算を試して動作確認
        _ = cp.array([1], dtype=cp.int32) * 2
        return True
    except Exception:
        return False


if __name__ == "__main__":
    # QApplication のインスタンス化
    app = QApplication(sys.argv)

    # 検出されたディスプレイの情報をコンソールに出力
    print("=== Display Mapping ===")
    screens = QGuiApplication.screens()
    for i, s in enumerate(screens):
        g = s.geometry()
        print(f"[{i}] {s.name()} : {g.width()}x{g.height()} at ({g.x()},{g.y()})")
    print("========================")

    # メインウィンドウの起動とイベントループ開始
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())