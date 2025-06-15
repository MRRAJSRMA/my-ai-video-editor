import moviepy.editor as mp
import librosa
import numpy as np
import os

def detect_silences(audio_path, silence_thresh=-40, min_silence_len=1.0):
    y, sr = librosa.load(audio_path)
    silence = librosa.effects.split(y, top_db=abs(silence_thresh))
    
    # Find non-silent parts
    segments = []
    for start, end in silence:
        start_sec = start / sr
        end_sec = end / sr
        if (end_sec - start_sec) > min_silence_len:
            segments.append((start_sec, end_sec))
    return segments

def auto_cut_video(video_path, output_path="auto_cut_output.mp4", silence_db=-40):
    video = mp.VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    video.audio.write_audiofile(audio_path, verbose=False, logger=None)

    keep_segments = detect_silences(audio_path, silence_thresh=silence_db)
    clips = []
    for start, end in keep_segments:
        clips.append(video.subclip(start, end))

    final = mp.concatenate_videoclips(clips)
    final.write_videofile(output_path, codec="libx264", audio_codec="aac")

    os.remove(audio_path)
