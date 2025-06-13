# distortion.py
import cv2
import numpy as np

def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (512, 512))

def generate_fisheye_image(image, output_size=512):
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    max_radius = min(cx, cy)
    fisheye_image = np.zeros_like(image)

    for y in range(output_size):
        for x in range(output_size):
            dx = x - output_size // 2
            dy = y - output_size // 2
            r = np.sqrt(dx**2 + dy**2)
            if r > output_size // 2:
                continue
            theta = r / (output_size // 2) * (np.pi / 2)
            phi = np.arctan2(dy, dx)
            src_x = cx + (theta / (np.pi / 2)) * max_radius * np.cos(phi)
            src_y = cy + (theta / (np.pi / 2)) * max_radius * np.sin(phi)
            ix, iy = int(src_x), int(src_y)
            if 0 <= ix < w-1 and 0 <= iy < h-1:
                fx, fy = src_x - ix, src_y - iy
                top = (1 - fx) * image[iy, ix] + fx * image[iy, ix+1]
                bot = (1 - fx) * image[iy+1, ix] + fx * image[iy+1, ix+1]
                color = (1 - fy) * top + fy * bot
                fisheye_image[y, x] = color.astype(np.uint8)
    return fisheye_image
