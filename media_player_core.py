# media_player_core.py
import numpy as np
import pygetwindow as gw
import win32gui
import win32con
import mss
import tkinter as tk
from tkinter import messagebox, simpledialog
from warp_engine import warp_image

def activate_window(hwnd):
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    win32gui.SetForegroundWindow(hwnd)

class MediaPlayer:
    def __init__(self):
        self.sct = mss.mss()

    def list_windows(self):
        return [w.title for w in gw.getWindows() if w.title.strip()]

    def select_window(self):
        windows = self.list_windows()
        if not windows:
            messagebox.showerror("エラー", "キャプチャ可能なウィンドウが見つかりません")
            return None

        root = tk.Tk()
        root.withdraw()
        msg = "起動中ウィンドウ\n\n" + "\n".join([f"[{i}] {title}" for i, title in enumerate(windows)])
        messagebox.showinfo("起動中ウィンドウ", msg)
        idx = simpledialog.askinteger("ウィンドウ選択", f"番号を入力してください（0～{len(windows)-1}）")
        root.destroy()

        if idx is None or idx < 0 or idx >= len(windows):
            messagebox.showerror("選択エラー", "無効な番号が選択されました")
            return None

        return gw.getWindowsWithTitle(windows[idx])[0]

    def capture_window(self, window):
        hwnd = window._hWnd
        rect = win32gui.GetWindowRect(hwnd)
        x, y, x2, y2 = rect
        w, h = x2 - x, y2 - y

        monitor = {"top": y, "left": x, "width": w, "height": h}
        img = np.array(self.sct.grab(monitor))[:, :, :3]
        return img

    def play_window_by_title(self, title):
        window = next((w for w in gw.getWindowsWithTitle(title) if title in w.title), None)
        if not window:
            print(f"ウィンドウ '{title}' が見つかりません")
            return

        activate_window(window._hWnd)
        print(f"選択ウィンドウ: {window.title}")

        while True:
            frame = self.capture_window(window)
            if frame is None:
                print("キャプチャ失敗")
                break
            corrected = warp_image(frame)
            import cv2
            cv2.imshow("Warped Window", corrected)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
