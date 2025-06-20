# view_environment.py（3D環境可視化ツール）
import matplotlib.pyplot as plt
from settings.config.environment_config import environment
import numpy as np


def draw_vector(ax, origin, direction, length=1.0, color='b', label=None):
    d = np.array(direction)
    o = np.array(origin)
    ax.quiver(*o, *(d / np.linalg.norm(d) * length), color=color, arrow_length_ratio=0.1)
    if label:
        ax.text(*(o + d * length * 1.1), label, color=color)


def plot_environment():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # スクリーン描画
    for screen in environment['screens']:
        c = np.array(screen['center'])
        n = np.array(screen['normal'])
        draw_vector(ax, c, n, length=0.5, color='green', label=screen['id'])
        ax.scatter(*c, color='lime', s=40)

    # プロジェクター描画
    for proj in environment['projectors']:
        p = np.array(proj['position'])
        draw_vector(ax, p, [1, 0, 0], length=0.5, color='red', label=proj['id'])
        ax.scatter(*p, color='red', s=30)

    # 鏡
    if 'mirror' in environment:
        m = np.array(environment['mirror']['center'])
        ax.scatter(*m, color='cyan', s=60, label='mirror')
        ax.text(*m, 'Mirror', color='cyan')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('360° Projection Environment')
    ax.grid(True)
    ax.set_box_aspect([1, 1, 0.7])
    plt.show()


if __name__ == '__main__':
    plot_environment()
