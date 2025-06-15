import cv2
import numpy as np
from rembg import remove
import moviepy.editor as mp
import os

def apply_body_segmentation(video_path, output_path="body_vfx_output.mp4", background_path=None):
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("temp_vfx.mp4", fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Remove background
        result = remove(frame)

        if background_path:
            bg = cv2.imread(background_path)
            bg = cv2.resize(bg, (width, height))
            mask = result[:, :, 3]
            mask_inv = cv2.bitwise_not(mask)

            fg = cv2.bitwise_and(result[:, :, :3], result[:, :, :3], mask=mask)
            bk = cv2.bitwise_and(bg, bg, mask=mask_inv)

            combined = cv2.add(fg, bk)
        else:
            combined = result[:, :, :3]

        out.write(combined)

    cap.release()
    out.release()

    # Add original audio back
    video = mp.VideoFileClip(video_path)
    final = mp.VideoFileClip("temp_vfx.mp4").set_audio(video.audio)
    final.write_videofile(output_path, codec="libx264", audio_codec="aac")

    os.remove("temp_vfx.mp4")
