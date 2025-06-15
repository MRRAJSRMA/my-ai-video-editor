# ----------- Text Overlay Functions ------------
from moviepy.editor import TextClip

def create_text_clip(text, duration, font="Arial-Bold", fontsize=50, color="white", position="bottom"):
    txt = TextClip(text, fontsize=fontsize, font=font, color=color)
    txt = txt.set_duration(duration).set_position(position).set_opacity(0.8)
    return txt

# ----------- Video Filters ---------------------
import cv2
import numpy as np

def apply_hdr_filter(frame):
    hdr = cv2.detailEnhance(frame, sigma_s=12, sigma_r=0.15)
    return hdr

def apply_warm_filter(frame):
    increase = np.full(frame.shape, (10, 0, -10), dtype=np.uint8)
    return cv2.add(frame, increase)

def apply_cool_filter(frame):
    increase = np.full(frame.shape, (-10, 0, 10), dtype=np.uint8)
    return cv2.add(frame, increase)
