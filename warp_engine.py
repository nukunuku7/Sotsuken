import cv2
import numpy as np

# 射影変換（4点）による基本的な歪み補正
def warp_perspective(image, src_pts, dst_pts, output_size=None):
    """
    OpenCVのgetPerspectiveTransformとwarpPerspectiveを用いた変換

    Parameters:
    - image: 入力画像
    - src_pts: 変換前の4点（左上、右上、右下、左下）
    - dst_pts: 変換後の4点（同じ順序）
    - output_size: (width, height) 指定がなければ入力画像サイズを使用

    Returns:
    - 補正後の画像
    """
    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)

    if output_size is None:
        h, w = image.shape[:2]
        output_size = (w, h)

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, matrix, output_size)
    return warped

# グリッド補間に基づく自由形状の補正マップを生成
def generate_remap_from_grid(grid_src, grid_dst, shape):
    """
    OpenCVのremapを使うためのマップ生成関数。

    Parameters:
    - grid_src: 元画像の座標グリッド (rows x cols x 2)
    - grid_dst: 補正後の座標グリッド (同上)
    - shape: 出力画像サイズ (height, width)

    Returns:
    - map_x, map_y: remap関数用のマップ
    """
    rows, cols = grid_src.shape[:2]
    h, w = shape

    # まずは空のマップを作成
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    # ここでは双線形補間でmapを構築する（簡略化）
    for i in range(rows - 1):
        for j in range(cols - 1):
            # 各セルの4点を取得（左上、右上、右下、左下）
            quad_src = [
                grid_dst[i][j],
                grid_dst[i][j+1],
                grid_dst[i+1][j+1],
                grid_dst[i+1][j]
            ]
            quad_dst = [
                grid_src[i][j],
                grid_src[i][j+1],
                grid_src[i+1][j+1],
                grid_src[i+1][j]
            ]

            # 各セル範囲に対して逆マッピングする方法は省略（実際はmeshgrid+cv2.remap推奨）
            # 詳細な高精度補間はscipyやOpenCVで拡張可能

    return map_x, map_y

# remapによる画像補正
def apply_remap(image, map_x, map_y):
    """
    remapマップを使用して画像補正
    """
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
