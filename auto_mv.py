import numpy as np
import pygame
import sounddevice as sd
import threading
import ctypes
import os
import time
from ctypes import Structure, windll, byref, c_long

# ウィンドウサイズ
WIDTH, HEIGHT = 800, 600

# ==== スクロールオフセット ====
scroll_offset_input = 0
scroll_offset_output = 0
scroll_speed = 20  # スクロールの速さ

# ==== デバイスインデックス ====
selected_inputs = []
selected_outputs = []

# ==== タイマー制御変数 ====
selection_complete_time = None
SELECTION_DELAY_SEC = 3

# ==== デバイス取得 ====
def list_devices(kind='input'):
    return [d for d in sd.query_devices() if (d['max_input_channels'] if kind == 'input' else d['max_output_channels']) > 0]

input_devices = list_devices('input')
output_devices = list_devices('output')

# ==== Pygame 初期化 ====
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("デバイス選択")
font = pygame.font.SysFont("meiryo", 18)  # UTF-8対応フォント（Windows日本語環境向け）


# ==== デバイス選択 UI 描画 ====
def draw_device_selection():
    screen.fill((30, 30, 30))
    title = font.render("録音（入力）デバイスを2つ選択 / 再生（出力）デバイスを2つ選択", True, (255, 255, 255))
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 20))

    # 入力デバイス描画
    for i, dev in enumerate(input_devices):
        y = 60 + i * 35 - scroll_offset_input
        if 60 <= y <= 290:  # 可視範囲だけ描画
            color = (0, 128, 255) if i in selected_inputs else (100, 100, 100)
            pygame.draw.rect(screen, color, (50, y, 700, 30))
            label = font.render(f"入力{i + 1}: {dev['name']}", True, (255, 255, 255))
            screen.blit(label, (60, y + 5))

    # 出力デバイス描画
    for j, dev in enumerate(output_devices):
        y = 320 + j * 35 - scroll_offset_output
        if 320 <= y <= HEIGHT - 40:  # 可視範囲だけ描画
            color = (0, 255, 128) if j in selected_outputs else (100, 100, 100)
            pygame.draw.rect(screen, color, (50, y, 700, 30))
            label = font.render(f"出力{j + 1}: {dev['name']}", True, (255, 255, 255))
            screen.blit(label, (60, y + 5))

    pygame.display.flip()

# ==== スクロール処理 ====
for event in pygame.event.get():
    if event.type == pygame.QUIT:
        pygame.quit()
        exit()

    elif event.type == pygame.MOUSEBUTTONDOWN:
        mx, my = pygame.mouse.get_pos()

        # マウスホイールでスクロール（Windowsならこれ）
        if event.button == 4:  # 上
            if my < 300:
                scroll_offset_input = max(scroll_offset_input - scroll_speed, 0)
            else:
                scroll_offset_output = max(scroll_offset_output - scroll_speed, 0)
        elif event.button == 5:  # 下
            if my < 300:
                scroll_offset_input += scroll_speed
            else:
                scroll_offset_output += scroll_speed

        # 入力選択判定
        for i in range(len(input_devices)):
                y = 60 + i * 35 - scroll_offset_input
                if 50 <= mx <= 750 and 60 <= y <= 90:
                    if selected_inputs == [i]:
                        selected_inputs.clear()
                    else:
                        selected_inputs = [i]

        # 出力選択判定
        for j in range(len(output_devices)):
                y = 320 + j * 35 - scroll_offset_output
                if 50 <= mx <= 750 and 320 <= y <= 350:
                    if selected_outputs == [j]:
                        selected_outputs.clear()
                    else:
                        selected_outputs = [j]

        # 両方選択されていたら、タイマーをセット（1回だけ）
        if len(selected_inputs) == 1 and len(selected_outputs) == 1 and selection_complete_time is None:
            selection_complete_time = time.time()

    elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_UP:
            scroll_offset_input = max(scroll_offset_input - scroll_speed, 0)
        elif event.key == pygame.K_DOWN:
            scroll_offset_input += scroll_speed
        elif event.key == pygame.K_w:
            scroll_offset_output = max(scroll_offset_output - scroll_speed, 0)
        elif event.key == pygame.K_s:
            scroll_offset_output += scroll_speed

# ==== デバイス監視スレッド ====
def monitor_devices():
    global input_devices, output_devices
    while True:
        time.sleep(1)
        input_devices = list_devices('input')
        output_devices = list_devices('output')

threading.Thread(target=monitor_devices, daemon=True).start()


# ==== UI 選択ループ ====
ui_running = True
while ui_running:
    draw_device_selection()

    # タイマー起動後に3秒経ったらUI終了
    if selection_complete_time:
        if time.time() - selection_complete_time > SELECTION_DELAY_SEC:
            ui_running = False
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
            
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            for i in range(len(input_devices)):
                if 50 <= mx <= 750 and 60 + i * 35 <= my <= 90 + i * 35:
                    if i in selected_inputs:
                        selected_inputs.remove(i)
                    elif len(selected_inputs) < 2:
                        selected_inputs.append(i)
            for j in range(len(output_devices)):
                if 50 <= mx <= 750 and 320 + j * 35 <= my <= 350 + j * 35:
                    if j in selected_outputs:
                        selected_outputs.remove(j)
                    elif len(selected_outputs) < 2:
                        selected_outputs.append(j)

    if len(selected_inputs) == 2 and len(selected_outputs) == 2:
        ui_running = False

pygame.display.quit()# UI非表示化

# ==== デバイス情報取得 ====
input_info1 = input_devices[selected_inputs[0]]
input_info2 = input_devices[selected_inputs[1]]
INPUT_DEVICE_INDEX = input_info1['index']
SR = int(input_info1['default_samplerate'])

# ==== 透過ウィンドウ用 ====
class RECT(Structure):
    _fields_ = [
        ("left", c_long),
        ("top", c_long),
        ("right", c_long),
        ("bottom", c_long)
    ]

class MONITORINFO(Structure):
    _fields_ = [
        ('cbSize', ctypes.c_ulong),
        ('rcMonitor', RECT),
        ('rcWork', RECT),
        ('dwFlags', ctypes.c_ulong)
    ]

def get_monitor_work_area(hwnd):
    monitor = windll.user32.MonitorFromWindow(hwnd, 2)
    info = MONITORINFO()
    info.cbSize = ctypes.sizeof(MONITORINFO)
    windll.user32.GetMonitorInfoW(monitor, byref(info))
    rc_work = info.rcWork
    rc_monitor = info.rcMonitor
    return (
        rc_work.left, rc_work.top,
        rc_work.right, rc_work.bottom,
        rc_monitor.left, rc_monitor.top,
        rc_monitor.right, rc_monitor.bottom
    )

# ==== 表示ウィンドウ（透明） ====
pygame.display.init()
temp_screen = pygame.display.set_mode((100, 100))
hwnd = pygame.display.get_wm_info()["window"]
left, top, right, bottom, *_ = get_monitor_work_area(hwnd)
pygame.display.quit()

VISUALIZER_HEIGHT = 200
SCREEN_WIDTH = right - left
WINDOW_Y = bottom - VISUALIZER_HEIGHT
os.environ['SDL_VIDEO_WINDOW_POS'] = f'{left},{WINDOW_Y}'
screen = pygame.display.set_mode((SCREEN_WIDTH, VISUALIZER_HEIGHT), pygame.NOFRAME)
pygame.display.set_caption("Visualizer")
hwnd = pygame.display.get_wm_info()["window"]
extended_style = windll.user32.GetWindowLongW(hwnd, -20)
windll.user32.SetWindowLongW(hwnd, -20, extended_style | 0x80000)
windll.user32.SetLayeredWindowAttributes(hwnd, 0x000000, 0, 0x1)

# ==== ビジュアライザー用パラメータ ====
BLOCK_SIZE = 2048
N_BARS = 128
BAR_HEIGHT = 160
BAR_WIDTH = screen.get_width() // N_BARS
buffer = np.zeros(BLOCK_SIZE, dtype=np.float32)
clock = pygame.time.Clock()

# ==== 録音処理 ====
def audio_callback(indata, frames, time_info, status):
    global buffer
    buffer = indata[:, 0]

stream = sd.InputStream(
    samplerate=SR,
    blocksize=BLOCK_SIZE,
    device=INPUT_DEVICE_INDEX,
    channels=1,
    dtype='float32',
    callback=audio_callback
)
stream.start()

# ==== 周波数ビン作成 ====
def create_custom_log_bins(sr, n_bars, linear_cutoff=1200, linear_ratio=0.4, min_freq=20):
    linear_bins = int(n_bars * linear_ratio)
    log_bins_count = n_bars - linear_bins

    linear_edges = np.linspace(min_freq, linear_cutoff, linear_bins + 1, endpoint=True)
    log_edges = np.logspace(np.log10(linear_cutoff), np.log10(sr / 2), log_bins_count + 1, endpoint=True)

    # 🔽 重複回避：linearの最後を除き、logは全て使用
    return np.concatenate((linear_edges[:-1], log_edges))

# ==== ログビン生成 ====
log_bins = create_custom_log_bins(SR, N_BARS)

# ==== スペクトル計算 ====
def get_freq_spectrum(audio, log_bins):
    windowed = audio * np.hanning(len(audio))
    fft = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(audio), 1 / SR)
    bar_heights = np.zeros(N_BARS)
    THRESHOLD_DB = -70
    BASE_GAIN_DB = 40

    for i in range(N_BARS):
        mask = (freqs >= log_bins[i]) & (freqs < log_bins[i + 1])
        # 🔽 maskが空のときでも無視せず、最小パワーで処理する
        power = np.mean(fft[mask]) if np.any(mask) else 1e-6
        db = 20 * np.log10(power + 1e-6)

        freq_center = (log_bins[i] + log_bins[i + 1]) / 2
        gain_adjust_db = np.interp(
            np.log10(freq_center),
            [np.log10(log_bins[0]), np.log10(log_bins[-1])],
            [0, -40]
        )
        total_gain = BASE_GAIN_DB + gain_adjust_db

        if db < THRESHOLD_DB:
            power = np.mean(fft[mask]) if np.any(mask) else 1e-6
        else:
            norm = ((db - total_gain) - THRESHOLD_DB) / (-THRESHOLD_DB)
            bar_heights[i] = np.clip(norm**2.0, 0, 1)

    return bar_heights

# ==== メインループ ====
running = True
while running:
    screen.fill((0, 0, 0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    bar_heights = get_freq_spectrum(buffer, log_bins)
    for i, mag in enumerate(bar_heights):
        h = int(mag * BAR_HEIGHT)
        x = i * BAR_WIDTH
        y = screen.get_height() - h
        red = int(255 * mag)
        blue = int(255 * (1 - mag))
        color = (red, 0, blue)
        pygame.draw.rect(screen, color, (x, y, BAR_WIDTH - 2, h))

    pygame.display.flip()
    clock.tick(60)

# ==== 終了処理 ====
stream.stop()
stream.close()
pygame.quit()
