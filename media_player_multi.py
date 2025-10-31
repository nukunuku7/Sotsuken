import sys
import argparse
import cv2
import numpy as np
import mss
from PyQt5.QtWidgets import QApplication, QLabel, QWidget
from PyQt5.QtGui import QImage, QPixmap, QGuiApplication
from PyQt5.QtCore import QTimer, Qt
from warp_engine import warp_image, prepare_warp


class DisplayWindow(QWidget):
    def __init__(self, source_screen, target_screen, mode, offset_x, virtual_size, fade_enabled=False):
        super().__init__()
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_DeleteOnClose)

        self.source_screen = source_screen
        self.target_screen = target_screen
        self.fade_enabled = fade_enabled
        self.mode = mode
        self.offset_x = offset_x
        self.virtual_size = virtual_size  # (total_width, height)

        geom_src = source_screen.geometry()
        geom_tgt = target_screen.geometry()
        self.setGeometry(geom_tgt)

        self.label = QLabel(self)
        self.label.setGeometry(0, 0, geom_tgt.width(), geom_tgt.height())

        # MSSによるキャプチャ設定（ソース全体キャプチャ）
        self.sct = mss.mss()
        self.mon = {
            "left": geom_src.x(),
            "top": geom_src.y(),
            "width": geom_src.width(),
            "height": geom_src.height()
        }

        # warp情報を初期化
        self.warp_info = prepare_warp(
            target_screen.name(),
            self.mode,
            (geom_tgt.width(), geom_tgt.height())
        )

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)
        self.showFullScreen()

    def update_frame(self):
        raw = np.array(self.sct.grab(self.mon))
        if raw is None or raw.size == 0:
            return

        frame = raw[:, :, :3]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 仮想ワイド画面対応：全体から自分の領域を切り出す
        total_w, total_h = self.virtual_size
        geom_tgt = self.target_screen.geometry()
        part_width = geom_tgt.width()
        x_start = int((self.offset_x / total_w) * frame.shape[1])
        x_end = int(((self.offset_x + part_width) / total_w) * frame.shape[1])
        sub_frame = frame[:, x_start:x_end]

        # warp適用
        warped = warp_image(sub_frame, warp_info=self.warp_info)
        if warped is None:
            return

        # 複数台ならフェード適用
        if warped is not None and self.fade_enabled:
            h, w = warped.shape[:2]
            fade = np.ones((h, w), dtype=np.float32)

            # warp_map 用はマスクから、perspective 用は矩形範囲から左右フェード
            if self.mode == "warp_map":
                mask = np.zeros((h, w), dtype=np.uint8)
                pts = np.array(self.warp_info.get("map_x", []), dtype=np.int32)
                if pts.size > 0:
                    cv2.fillPoly(mask, [pts], 255)
                    x_indices = np.where(mask > 0)[1]
                    if len(x_indices) > 0:
                        x_min, x_max = x_indices.min(), x_indices.max()
                        blend_w = max(int((x_max - x_min) * 0.1), 1)
                        for x in range(blend_w):
                            alpha = x / float(blend_w)
                            fade[:, x_min + x] *= alpha
                            fade[:, x_max - x] *= alpha
                    fade[mask == 0] = 0.0
            elif self.mode == "perspective":
                pts = np.array(self.warp_info.get("matrix", []), dtype=np.float32)
                if pts.size >= 4:
                    x_coords = pts[:,0]
                    x_min, x_max = int(x_coords.min()), int(x_coords.max())
                    blend_w = max(int((x_max - x_min) * 0.1), 1)
                    for x in range(blend_w):
                        alpha = x / float(blend_w)
                        fade[:, x_min + x] *= alpha
                        fade[:, x_max - x] *= alpha

            warped = (warped.astype(np.float32) * fade[..., None]).astype(np.uint8)

        h, w, ch = warped.shape
        bytes_per_line = ch * w
        qt_image = QImage(warped.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qt_image))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="キャプチャ元ディスプレイ名")
    parser.add_argument("--targets", nargs="+", required=True, help="補正出力先ディスプレイ名（複数可）")
    parser.add_argument("--mode", choices=["perspective", "warp_map"], default="perspective")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    screens = {s.name(): s for s in QGuiApplication.screens()}

    if args.source not in screens:
        print(f"❌ ソースディスプレイが見つかりません: {args.source}")
        sys.exit(1)

    source_screen = screens[args.source]
    total_width = sum(screens[n].geometry().width() for n in args.targets if n in screens)
    max_height = max(screens[n].geometry().height() for n in args.targets if n in screens)
    virtual_size = (total_width, max_height)

    windows = []
    offset_x = 0

    for name in args.targets:
        if name not in screens:
            print(f"⚠️ ターゲットディスプレイが見つかりません: {name}")
            continue

        target_screen = screens[name]
        print(f"🎥 {args.source} → {name} に補正出力します (モード: {args.mode}) [offset={offset_x}]")

        fade_enabled = len(args.targets) > 1
        window = DisplayWindow(
            source_screen, target_screen, args.mode,
            offset_x, virtual_size, fade_enabled=fade_enabled
        )
        windows.append(window)
        offset_x += target_screen.geometry().width()

    if not windows:
        print("❌ 出力ディスプレイがありません。終了します。")
        sys.exit(1)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
