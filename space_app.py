import gradio as gr
from app import full_ai_video_pipeline  # Ye function app.py me banega

def process(video, background, caption, zoom, body):
    return full_ai_video_pipeline(
        video_path=video,
        background=background,
        caption=caption,
        zoom=zoom,
        body_effect=body
    )

iface = gr.Interface(
    fn=process,
    inputs=[
        gr.Video(label="Upload Video"),
        gr.Radio(["None", "Beach", "City", "Blurred"], label="Background"),
        gr.Checkbox(label="Add Auto Captions"),
        gr.Checkbox(label="Apply Zoom + Shake"),
        gr.Checkbox(label="Enable Body VFX")
    ],
    outputs=gr.Video(label="Edited AI Video"),
    title="🎬 AI Video Editor",
    description="Upload a video and apply cool effects using AI!"
)

iface.launch()
