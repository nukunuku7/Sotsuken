import numpy as np
import pygame
import sounddevice as sd
import os
import ctypes
from ctypes import Structure, windll, byref, c_long

# ======= 定数設定 =======
SR = 44100
BLOCK_SIZE = 2048
N_BARS = 128
BAR_HEIGHT = 120

# ======= 作業領域の取得（複数ディスプレイ対応） =======
class RECT(Structure):
    _fields_ = [("left", c_long), ("top", c_long), ("right", c_long), ("bottom", c_long)]

class MONITORINFO(Structure):
    _fields_ = [
        ('cbSize', ctypes.c_ulong),
        ('rcMonitor', RECT),
        ('rcWork', RECT),
        ('dwFlags', ctypes.c_ulong)
    ]

def get_monitor_work_area(hwnd):
    monitor = windll.user32.MonitorFromWindow(hwnd, 2)  # MONITOR_DEFAULTTONEAREST
    info = MONITORINFO()
    info.cbSize = ctypes.sizeof(MONITORINFO)
    windll.user32.GetMonitorInfoW(monitor, byref(info))
    rc_work = info.rcWork
    rc_monitor = info.rcMonitor
    return (rc_work.left, rc_work.top, rc_work.right, rc_work.bottom,
            rc_monitor.left, rc_monitor.top, rc_monitor.right, rc_monitor.bottom)

# ======= 仮ウィンドウ作成（モニター情報取得のため） =======
pygame.init()
temp_screen = pygame.display.set_mode((100, 100))
hwnd = pygame.display.get_wm_info()["window"]
(left, top, right, bottom, mon_left, mon_top, mon_right, mon_bottom) = get_monitor_work_area(hwnd)
pygame.display.quit()

# ======= 表示サイズと位置設定 =======
VISUALIZER_HEIGHT = 200
SCREEN_WIDTH = right - left
WINDOW_Y = bottom - VISUALIZER_HEIGHT

# ======= 本ウィンドウ作成（透明・枠なし） =======
os.environ['SDL_VIDEO_WINDOW_POS'] = f'{left},{WINDOW_Y}'
screen = pygame.display.set_mode((SCREEN_WIDTH, VISUALIZER_HEIGHT), pygame.NOFRAME)
pygame.display.set_caption("")
hwnd = pygame.display.get_wm_info()["window"]

# ======= 透過処理（黒背景を透明に） =======
extended_style = windll.user32.GetWindowLongW(hwnd, -20)
windll.user32.SetWindowLongW(hwnd, -20, extended_style | 0x80000)
windll.user32.SetLayeredWindowAttributes(hwnd, 0x000000, 0, 0x1)

# ======= 周波数ビンの生成（線形+対数のハイブリッド） =======
def create_custom_log_bins(sr, n_bars, linear_cutoff=500, linear_ratio=0.2, min_freq=30):
    linear_bins = int(n_bars * linear_ratio)
    log_bins_count = n_bars - linear_bins
    linear_edges = np.linspace(min_freq, linear_cutoff, linear_bins + 1)
    log_edges = np.logspace(np.log10(linear_cutoff), np.log10(sr / 2), log_bins_count + 1)
    return np.concatenate((linear_edges, log_edges[1:]))

log_bins = create_custom_log_bins(SR, N_BARS, linear_cutoff=300, linear_ratio=0.2)

# ======= スペクトル計算 =======
def get_freq_spectrum(audio, log_bins):
    windowed = audio * np.hanning(len(audio))
    fft = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(audio), 1 / SR)
    bar_heights = np.zeros(N_BARS)
    THRESHOLD_DB = -80

    for i in range(N_BARS):
        mask = (freqs >= log_bins[i]) & (freqs < log_bins[i + 1])
        if np.any(mask):
            power = np.mean(fft[mask])
            db = 20 * np.log10(power + 1e-6)
            if db < THRESHOLD_DB:
                bar_heights[i] = 0
            else:
                norm = (db - THRESHOLD_DB) / (-THRESHOLD_DB)
                bar_heights[i] = np.clip(norm**2.0, 0, 1)
    return bar_heights, freqs[np.argmax(fft)]

# ======= 録音コールバック =======
buffer = np.zeros(BLOCK_SIZE, dtype=np.float32)
def audio_callback(indata, frames, time_info, status):
    global buffer
    buffer = indata[:, 0]

# ======= サウンドストリーム開始 =======
stream = sd.InputStream(channels=1, samplerate=SR, blocksize=BLOCK_SIZE, callback=audio_callback)
stream.start()

# ======= 棒幅設定 =======
BAR_WIDTH = screen.get_width() // N_BARS
clock = pygame.time.Clock()

# ======= メインループ =======
running = True
while running:
    screen.fill((0, 0, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    bar_heights, peak_freq = get_freq_spectrum(buffer, log_bins)

    for i, mag in enumerate(bar_heights):
        h = int(mag * BAR_HEIGHT)
        x = i * BAR_WIDTH
        y = screen.get_height() - h
        red = int(255 * mag)
        blue = int(255 * (1 - mag))
        color = (red, 0, blue)
        pygame.draw.rect(screen, color, (x, y, BAR_WIDTH - 2, h))

    pygame.display.flip()
    clock.tick(120)

# ======= 終了処理 =======
stream.stop()
stream.close()
pygame.quit()
