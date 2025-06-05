import numpy as np
import pygame
import sounddevice as sd
import time
import colorsys
import random
import os

# ======= 定数設定 =======
SR = 44100
BLOCK_SIZE = 2048
N_BARS = 128
BAR_HEIGHT = 120
BAR_WIDTH = 1200 // N_BARS

# ======= Pygame初期化 =======
os.environ['SDL_VIDEO_CENTERED'] = '1'
pygame.init()
screen = pygame.display.set_mode((1200, 200))
pygame.display.set_caption("リアルタイムAudioビジュアライザー")
clock = pygame.time.Clock()

# ======= ログスケール周波数ビン作成 =======
log_bins = np.logspace(np.log10(20), np.log10(SR / 2), N_BARS + 1, base=10)

# ======= カスタムログスケールビン作成関数 =======
def create_custom_log_bins(sr, n_bars, linear_cutoff=500, linear_ratio=0.2, min_freq=30):
    """
    線形+対数のハイブリッドlog_binsを生成
    - linear_cutoff: 線形→対数の境界周波数
    - linear_ratio: 線形部分のバー数割合（0.0〜1.0）
    """
    linear_bins = int(n_bars * linear_ratio)
    log_bins_count = int(n_bars - linear_bins)

    # 線形スケール部分（min_freq〜linear_cutoff）
    linear_edges = np.linspace(min_freq, linear_cutoff, linear_bins + 1)

    # 対数スケール部分（linear_cutoff〜Nyquist）
    log_edges = np.logspace(np.log10(linear_cutoff), np.log10(sr / 2), log_bins_count + 1)

    # 線形の最後と対数の最初が同じになるので、重複を除いて結合
    full_bins = np.concatenate((linear_edges, log_edges[1:]))

    return full_bins


# ======= 周波数スペクトルの計算 =======
def get_freq_spectrum(audio, log_bins):
    windowed = audio * np.hanning(len(audio))
    fft = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(audio), 1 / SR)
    bar_heights = np.zeros(N_BARS)

    THRESHOLD_DB = -80  # 表示の最小しきい値（より静かな音にも対応）

    for i in range(N_BARS):
        mask = (freqs >= log_bins[i]) & (freqs < log_bins[i + 1])
        if np.any(mask):
            power = np.mean(fft[mask])
            db = 20 * np.log10(power + 1e-6)

            if db < THRESHOLD_DB:
                bar_heights[i] = 0
            else:
                normalized = (db - THRESHOLD_DB) / (-THRESHOLD_DB)
                bar_heights[i] = np.clip(normalized**2.0, 0, 1)

    return bar_heights, freqs[np.argmax(fft)]


# ======= 録音コールバック =======
buffer = np.zeros(BLOCK_SIZE, dtype=np.float32)

def audio_callback(indata, frames, time_info, status):
    global buffer
    buffer = indata[:, 0]

# ======= ストリーム開始 =======
stream = sd.InputStream(
    channels=1,
    samplerate=SR,
    blocksize=BLOCK_SIZE,
    callback=audio_callback
)
stream.start()

# N_BARS に応じて自動的に線形／対数分割
log_bins = create_custom_log_bins(SR, N_BARS, linear_cutoff=300, linear_ratio=0.2)
BAR_WIDTH = screen.get_width() // N_BARS  # 幅を自動で算出

# ======= メインループ =======
running = True
while running:
    screen.fill((0, 0, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 周波数スペクトル取得（1回だけ）
    bar_heights, peak_freq = get_freq_spectrum(buffer, log_bins)

    # 棒グラフ描画
    for i, mag in enumerate(bar_heights):
        h = int(mag * BAR_HEIGHT)
        x = i * BAR_WIDTH
        y = screen.get_height() - BAR_HEIGHT + BAR_HEIGHT - h

        red = int(255 * mag)
        green = 0
        blue = int(255 * (1 - mag))
        color = (red, green, blue)
        pygame.draw.rect(screen, color, (x, y, BAR_WIDTH - 2, h))

    pygame.display.flip()
    clock.tick(120)  # 120 FPS

# ======= 終了処理 =======
stream.stop()
stream.close()
pygame.quit()
