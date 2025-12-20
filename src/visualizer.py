import colorsys
import random
import librosa
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from tqdm import tqdm

FPS = 30
VIDEO_RES = (854, 480)
PALETTE = [(255, 230, 0), (0, 255, 255), (255, 0, 255)]

def generate_visualizer_frames(audio_path, beat_times, frame_dir):
    y, sr = librosa.load(audio_path, sr=None)
    hop = sr // FPS
    n_frames = int(len(y) / hop) + 1
    
    # Pre-calculate STFT
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    
    # Momentum variables
    smooth_bass = 0.0
    smooth_amp = 0.0
    prev_img = None 

    # Safety Margins
    center_x, center_y = VIDEO_RES[0] // 2, VIDEO_RES[1] // 2
    base_radius = 110  # Reduced slightly to give more room for peaks
    max_room = (min(VIDEO_RES) // 2) - base_radius - 20 # Max height before clipping

    for i in tqdm(range(n_frames)):
        frame_time = i / FPS
        start_sample = i * hop
        end_sample = min(start_sample + hop, len(y))
        
        # 1. Amplitude with Noise Floor
        # Adding 0.01 ensures there is always a tiny bit of movement
        raw_amp = np.max(np.abs(y[start_sample:end_sample])) if start_sample < len(y) else 0.01
        
        if raw_amp > smooth_amp:
            smooth_amp = raw_amp
        else:
            smooth_amp *= 0.9 # Slightly slower decay for "smoother" quiet parts
            
        # 2. Bass & Spectrum
        stft_idx = np.searchsorted(librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=512), frame_time)
        
        # 3. Canvas setup
        img = Image.new("RGBA", VIDEO_RES, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Dynamic Zoom based on bass (clamped)
        zoom = 1.0 + (min(smooth_amp, 0.5) * 0.15) 
        
        n_bars = 72
        for b in range(n_bars):
            angle = b * (2 * np.pi / n_bars) - np.pi / 2
            f_idx = int((b / n_bars) * (S.shape[0] // 4)) 
            
            # --- THE FIX: Non-Linear Scaling ---
            # 1. Get raw magnitude
            raw_mag = S[f_idx, stft_idx] if stft_idx < S.shape[1] else 0
            
            # 2. Add a tiny bit of random jitter (noise) for the "alive" look in quiet parts
            jitter = random.uniform(0, 0.02)
            
            # 3. Logarithmic scaling: log1p(x) makes small values larger and large values smaller
            # We multiply by smooth_amp so the whole visualizer still fades with the song
            val = np.log1p(raw_mag * 10 + jitter) * smooth_amp
            
            # 4. Map to screen pixels and clamp to safety room
            bar_height = min(val * 40 * zoom, max_room)
            # Ensure a minimum visible bar even in total silence
            bar_height = max(bar_height, 4) 

            # Color Logic
            color_idx = (b / n_bars) * (len(PALETTE) - 1)
            c1, c2 = PALETTE[int(color_idx)], PALETTE[min(int(color_idx) + 1, len(PALETTE)-1)]
            mix = color_idx - int(color_idx)
            r = int(c1[0] * (1-mix) + c2[0] * mix)
            g = int(c1[1] * (1-mix) + c2[1] * mix)
            bl = int(c1[2] * (1-mix) + c2[2] * mix)

            # Draw Bars
            x0 = center_x + (base_radius * zoom) * np.cos(angle)
            y0 = center_y + (base_radius * zoom) * np.sin(angle)
            x1 = center_x + (base_radius * zoom + bar_height) * np.cos(angle)
            y1 = center_y + (base_radius * zoom + bar_height) * np.sin(angle)
            
            draw.line([x0, y0, x1, y1], fill=(r, g, bl, 255), width=int(4 * zoom))

        img.save(frame_dir / f"frame_{i:05d}.png")