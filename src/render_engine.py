import subprocess
import numpy as np
import librosa
from PIL import Image, ImageDraw
from tqdm import tqdm
from pathlib import Path

def get_frame_count(audio_path, fps):
    y_duration = librosa.get_duration(path=str(audio_path))
    return int(y_duration * fps) + 1

def stream_visualizer_frames(audio_path, fps=30, resolution=(854, 480)):
    y, sr = librosa.load(audio_path, sr=None)
    hop = sr // fps
    n_frames = int(len(y) / hop) + 1
    
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freq_times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=512)
    
    smooth_amp = 0.01
    center_x, center_y = resolution[0] // 2, resolution[1] // 2
    base_radius = 110
    max_room = (min(resolution) // 2) - base_radius - 20
    palette = [(255, 230, 0), (0, 255, 255), (255, 0, 255)]

    for i in range(n_frames):
        frame_time = i / fps
        stft_idx = np.searchsorted(freq_times, frame_time)
        
        start_sample = i * hop
        end_sample = min(start_sample + hop, len(y))
        # Get the max volume of this specific frame
        raw_amp = np.max(np.abs(y[start_sample:end_sample])) if start_sample < len(y) else 0.01
        
        # Update the momentum (Instant hit, slow decay)
        if raw_amp > smooth_amp:
            smooth_amp = raw_amp
        else:
            smooth_amp *= 0.92  # This makes the bars "glide" down
        
        # 1. CRITICAL: The canvas MUST be RGBA with 0 alpha (fully transparent)
        img = Image.new("RGBA", resolution, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        n_bars = 72
        for b in range(n_bars):
            angle = b * (2 * np.pi / n_bars) - np.pi / 2
            f_idx = int((b / n_bars) * (S.shape[0] // 4)) 
            raw_mag = S[f_idx, stft_idx] if stft_idx < S.shape[1] else 0
            
            # Viral Scaling: Logarithmic + Noise Floor
            val = np.log1p(raw_mag * 12 + 0.02) * smooth_amp
            bar_height = max(4, min(val * 50, max_room))

            # Color Gradient
            color_idx = (b / n_bars) * (len(palette) - 1)
            c1, c2 = palette[int(color_idx)], palette[min(int(color_idx) + 1, len(palette)-1)]
            mix = color_idx - int(color_idx)
            color = (
                int(c1[0]*(1-mix) + c2[0]*mix),
                int(c1[1]*(1-mix) + c2[1]*mix),
                int(c1[2]*(1-mix) + c2[2]*mix), 255
            )

            x0 = center_x + base_radius * np.cos(angle)
            y0 = center_y + base_radius * np.sin(angle)
            x1 = center_x + (base_radius + bar_height) * np.cos(angle)
            y1 = center_y + (base_radius + bar_height) * np.sin(angle)
            draw.line([x0, y0, x1, y1], fill=color, width=4)

        # 2. FORCE RAW BYTES: Do NOT use .convert("RGB")
        yield img.tobytes()

def run_integrated_render(audio_path, ass_file, bg_source, output_path, resolution=(854, 480), fps=30):
    res_str = f"{resolution[0]}x{resolution[1]}"
    ass_path_fixed = str(ass_file.absolute()).replace("\\", "/").replace(":", "\\:")
    is_video = bg_source.suffix.lower() == ".mp4"
    total_frames = get_frame_count(audio_path, fps)

    cmd = ["ffmpeg", "-y", "-loglevel", "error"]

    # Input 0: Background
    if is_video:
        cmd += ["-stream_loop", "-1", "-i", str(bg_source)]
    else:
        cmd += ["-framerate", str(fps), "-loop", "1", "-i", str(bg_source)]

    # Input 1: The Pipe
    cmd += [
        "-f", "rawvideo", 
        "-vcodec", "rawvideo", 
        "-s", res_str,
        "-pix_fmt", "rgba", # Must match img.tobytes()
        "-framerate", str(fps), 
        "-i", "-"
    ]

    # Input 2: Audio
    cmd += ["-i", str(audio_path)]

    # --- THE FILTER COMPLEX FIX ---
    # We use [1:v]format=rgba to ensure FFmpeg knows the pipe has an alpha channel
    cmd += [
        "-filter_complex",
        f"[0:v]fps={fps},scale={resolution[0]}:{resolution[1]},setsar=1[bg];"
        f"[1:v]format=rgba,scale={resolution[0]}:{resolution[1]}[vis];"
        f"[bg][vis]overlay=0:0:shortest=1[v];"
        f"[v]ass='{ass_path_fixed}'[outv]",
        "-map", "[outv]", "-map", "2:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-preset", "veryfast", "-c:a", "aac", "-shortest", str(output_path)
    ]

    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    try:
        with tqdm(total=total_frames, desc=f"🎬 Rendering {output_path.name}") as pbar:
            for frame_bytes in stream_visualizer_frames(audio_path, fps, resolution):
                process.stdin.write(frame_bytes)
                pbar.update(1)
    finally:
        process.stdin.close()
        process.wait()