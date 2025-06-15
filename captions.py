import moviepy.editor as mp
import whisper

# Load Whisper AI model
model = whisper.load_model("base")

def extract_captions(video_path):
    result = model.transcribe(video_path)
    return result["text"]

def overlay_caption(video_path, caption_text, output_path="captioned_output.mp4"):
    video = mp.VideoFileClip(video_path)
    txt_clip = mp.TextClip(caption_text, fontsize=40, color='white', bg_color='black')
    txt_clip = txt_clip.set_duration(video.duration).set_position(("bottom")).set_opacity(0.7)

    final = mp.CompositeVideoClip([video, txt_clip])
    final.write_videofile(output_path, codec="libx264", audio_codec="aac")
