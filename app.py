from captions import extract_captions, overlay_caption
from effects import apply_shake, apply_zoom, apply_glitch
import gradio as gr
import cv2
import tempfile
import os
import mediapipe as mp
import numpy as np
from rembg import remove

def process_video(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(3))
    height = int(cap.get(4))

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

            cx = int((left_shoulder.x + right_shoulder.x) / 2 * width)
            cy = int((left_shoulder.y + right_shoulder.y) / 2 * height)
            zoom = 1.3

            x1 = int(cx - width // (2 * zoom))
            y1 = int(cy - height // (2 * zoom))
            x2 = int(cx + width // (2 * zoom))
            y2 = int(cy + height // (2 * zoom))

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            cropped = frame[y1:y2, x1:x2]
            resized = cv2.resize(cropped, (width, height))

            # Background removal
            bg_removed = remove(resized)
            out.write(bg_removed)
        else:
            out.write(frame)

    cap.release()
    out.release()
    return temp_file.name

gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload Video"),
    outputs=gr.Video(label="Processed Video"),
    title="AI Video Editor - Phase 1",
    description="Smart body detection, zoom effect, and background removal using AI.",
).launch()
