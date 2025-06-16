# media_player.py
import cv2
import numpy as np
import pygetwindow as gw
import win32gui
import mss
from warp_engine import warp_image  # 歪み補正関数をwarp_engine.pyから呼び出す

class MediaPlayer:
    def __init__(self):
        self.sct = mss.mss()

    def list_windows(self):
        return [w.title for w in gw.getWindows() if w.title.strip()]

    def select_window(self):
        windows = self.list_windows()
        if not windows:
            print("キャプチャ可能なウィンドウが見つかりません")
            return None

        for i, title in enumerate(windows):
            print(f"[{i}] {title}")
        try:
            idx = int(input("キャプチャするウィンドウ番号を選択してください: "))
            return gw.getWindowsWithTitle(windows[idx])[0]
        except Exception as e:
            print(f"選択エラー: {e}")
            return None

    def capture_window(self, window):
        hwnd = window._hWnd
        rect = win32gui.GetWindowRect(hwnd)
        x, y, x2, y2 = rect
        w, h = x2 - x, y2 - y

        monitor = {"top": y, "left": x, "width": w, "height": h}
        img = np.array(self.sct.grab(monitor))[:, :, :3]  # BGRA → BGR

        return img

    def play_window(self):
        window = self.select_window()
        if not window:
            return

        print(f"選択ウィンドウ: {window.title}")
        while True:
            frame = self.capture_window(window)
            if frame is None:
                print("キャプチャ失敗")
                break

            # 歪み補正（warp_engine.py）
            corrected = warp_image(frame)

            cv2.imshow("Warped Window", corrected)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    player = MediaPlayer()
    player.play_window()
