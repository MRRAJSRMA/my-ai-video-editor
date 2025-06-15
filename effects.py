import cv2
import random
import numpy as np

def apply_shake(frame, strength=10):
    rows, cols, _ = frame.shape
    dx = random.randint(-strength, strength)
    dy = random.randint(-strength, strength)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, M, (cols, rows))

def apply_zoom(frame, zoom_factor=1.2):
    h, w = frame.shape[:2]
    nh, nw = int(h / zoom_factor), int(w / zoom_factor)
    y1 = (h - nh) // 2
    x1 = (w - nw) // 2
    zoomed = frame[y1:y1+nh, x1:x1+nw]
    return cv2.resize(zoomed, (w, h))

def apply_glitch(frame, strength=5):
    glitched = frame.copy()
    h, w, _ = frame.shape
    for i in range(3):
        dx = random.randint(-strength, strength)
        glitched[:, :, i] = np.roll(glitched[:, :, i], dx, axis=1)
    return glitched
