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
RIPPLE_INTERVAL = 0.1  # 秒間隔で波紋生成
BAR_WIDTH = 1200 // N_BARS

# ======= Pygame初期化 =======
os.environ['SDL_VIDEO_CENTERED'] = '1'
pygame.init()
screen = pygame.display.set_mode((1200, 900))
pygame.display.set_caption("リアルタイムAudioビジュアライザー")
clock = pygame.time.Clock()

# ======= ログスケール周波数ビン作成 =======
log_bins = np.logspace(np.log10(20), np.log10(SR / 2), N_BARS + 1, base=10)

# ======= 流星情報 =======
meteors = []
last_meteor_time = 0
CENTER = (screen.get_width() // 2, screen.get_height() // 2)


def create_meteor():
    cx, cy = CENTER
    angle = random.uniform(0, 2 * np.pi)
    speed_init = 250  # 初速
    acc = 1000        # 加速度
    return {
        "pos": [cx, cy],
        "angle": angle,
        "speed": speed_init,
        "acc": acc,
        "start": time.time(),
        "duration": 3.0,
        "trail": [],
        "finished": False
    }


def out_of_screen(pos):
    x, y = pos
    return x < -100 or x > screen.get_width() + 100 or y < -100 or y > screen.get_height() + 100

# ======= 周波数スペクトルの計算 =======
def get_freq_spectrum(audio):
    windowed = audio * np.hanning(len(audio))
    fft = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(audio), 1 / SR)
    bar_heights = np.zeros(N_BARS)
    for i in range(N_BARS):
        mask = (freqs >= log_bins[i]) & (freqs < log_bins[i + 1])
        if np.any(mask):
            power = np.mean(fft[mask])
            db = 20 * np.log10(power + 1e-6)
            bar_heights[i] = np.clip((db + 80) / 80, 0, 1)
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

# ======= メインループ =======
running = True
while running:
    screen.fill((0, 0, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    spectrum, peak_freq = get_freq_spectrum(buffer)

    # 棒グラフ描画
    for i, mag in enumerate(spectrum):
        h = int(mag * BAR_HEIGHT)
        x = i * BAR_WIDTH
        y = screen.get_height() - BAR_HEIGHT + BAR_HEIGHT - h

        red = int(255 * mag)
        green = 0
        blue = int(255 * (1 - mag))
        color = (red, green, blue)
        pygame.draw.rect(screen, color, (x, y, BAR_WIDTH - 2, h))

    # === 流星生成 ===
    now = time.time()
    if now - last_meteor_time > RIPPLE_INTERVAL:
        meteors.append(create_meteor())
        last_meteor_time = now

    # === 流星描画 ===
    for m in meteors:
        elapsed = now - m["start"]

        # 加速移動
        m["speed"] += m["acc"] * (1 / 60)
        dx = np.cos(m["angle"]) * m["speed"] * (1 / 60)
        dy = np.sin(m["angle"]) * m["speed"] * (1 / 60)
        m["pos"][0] += dx
        m["pos"][1] += dy

        # トレイルを記録
        m["trail"].append(tuple(m["pos"]))

        # 端に到達したらフラグを立てる
        if not m["finished"] and out_of_screen(m["pos"]):
            m["finished"] = True

        # 終了処理：トレイルがなくなったら削除
        if m["finished"] and len(m["trail"]) == 0:
            continue

        # フェードアウト（後方から消す）
        if m["finished"] and len(m["trail"]) > 0:
            m["trail"].pop(0)

        # 描画
        surf = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        for i, pos in enumerate(m["trail"]):
            alpha = int(255 * (i / len(m["trail"])) if len(m["trail"]) > 0 else 0)
            pygame.draw.circle(surf, (180, 220, 255, alpha), (int(pos[0]), int(pos[1])), 2)
        # 星本体
        if not m["finished"]:
            pygame.draw.circle(surf, (180, 220, 255, 255), (int(m["pos"][0]), int(m["pos"][1])), 2)
        screen.blit(surf, (0, 0))

    pygame.display.flip()
    clock.tick(60)

# ======= 終了処理 =======
stream.stop()
stream.close()
pygame.quit()
