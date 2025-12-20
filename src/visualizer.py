from tqdm import tqdm
import colorsys
import random
import librosa
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

FPS = 30
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONTSIZE = 60
VIDEO_RES = (854, 480)

# -------------------
# Visualizer generation
# -------------------

def generate_visualizer_frames(audio_path, beat_times, frame_dir):
    y, sr = librosa.load(audio_path, sr=None)
    hop = sr // FPS
    n_frames = int(len(y) / hop) + 1
    print(f"[Visualizer] Generating {n_frames} frames...")

    max_amp = np.max(np.abs(y)) + 1e-6

    n_fft = 2048
    stft_hop = 512
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=stft_hop))
    stft_times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=stft_hop)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    font = ImageFont.truetype(FONT, FONTSIZE)
    
    for i in tqdm(range(n_frames)):
        frame_time = i / FPS
        is_beat = any(abs(frame_time - b) < 0.08 for b in beat_times)
        beat_intensity = 2.0 if is_beat else 1.0

        start_sample = i * hop
        end_sample = min(start_sample + hop, len(y))
        frame_audio = y[start_sample:end_sample]
        amp = np.max(np.abs(frame_audio)) / max_amp

        idx = np.searchsorted(stft_times, frame_time)
        spectrum = np.zeros(len(freqs))
        if 0 < idx < S.shape[1]:
            spectrum = np.mean(S[:, max(0, idx-2):idx+3], axis=1)
        max_spec = spectrum.max() + 1e-6

        img = Image.new("RGBA", VIDEO_RES, (0,0,0,0))
        draw = ImageDraw.Draw(img)

        center_x = VIDEO_RES[0] // 2
        center_y = VIDEO_RES[1] // 2
        n_bars = 60
        max_radius = min(VIDEO_RES) // 2 - 30

        # Bass boost for low freq
        bass = np.mean(spectrum[:100]) / max_spec

        for b in range(n_bars):
            angle = b * (2 * np.pi / n_bars) - np.pi / 2
            freq_idx = int((b / n_bars) * len(freqs))
            mag = (spectrum[freq_idx] / max_spec) ** 0.6
            height = mag * max_radius * amp * beat_intensity * (1 + bass)

            hue = (b / n_bars * 360 + frame_time * 50) % 360
            sat = 0.95 + 0.05 * mag

            # 10-layer glow
            for glow in range(10, 0, -1):
                radius = height + glow * 15
                alpha = int(255 * (0.9 / glow))
                r, g, b = colorsys.hsv_to_rgb(hue / 360, sat, 1.0)
                color = (int(r*255), int(g*255), int(b*255), alpha)
                thickness = 6 + glow
                x0 = center_x + (radius - thickness) * np.cos(angle)
                y0 = center_y + (radius - thickness) * np.sin(angle)
                x1 = center_x + (radius + thickness) * np.cos(angle)
                y1 = center_y + (radius + thickness) * np.sin(angle)
                draw.line([x0, y0, x1, y1], fill=color, width=thickness * 2)

        # Extra bass ring on beat
        if is_beat:
            for r in range(3):
                ring = 100 + r*30 + bass*200
                draw.ellipse([center_x - ring, center_y - ring, center_x + ring, center_y + ring],
                             outline=(255,100,255,200), width=10)

        # More particles
        if is_beat:
            for _ in range(50):
                p_angle = random.uniform(0, 2*np.pi)
                dist = random.uniform(0, max_radius * (1 + bass))
                px = center_x + dist * np.cos(p_angle)
                py = center_y + dist * np.sin(p_angle)
                size = random.randint(4,12)
                draw.ellipse([px-size, py-size, px+size, py+size], fill=(255,255,255,220))

        # Thick pulsing core
        pulse = 80 + 60 * amp * beat_intensity + bass*100
        draw.ellipse([center_x - pulse, center_y - pulse, center_x + pulse, center_y + pulse],
                     outline=(0,255,255,255), width=12)

        # Enhanced waveform
        wave_height = 80 * amp * beat_intensity
        n_points = VIDEO_RES[0] * 2
        if len(frame_audio) > 1:
            indices = np.linspace(0, len(frame_audio)-1, n_points, dtype=int)
            wave = frame_audio[indices] * wave_height
            points = [(x//2, center_y + int(wave[x])) for x in range(n_points)]
            draw.line(points, fill=(150,255,255,240), width=4)

        if amp < 0.05:
            text = "Quiet"
            bbox = draw.textbbox((0,0), text, font=font)
            w = bbox[2] - bbox[0]
            draw.text((center_x - w//2, center_y - 30), text, fill=(180,180,180,255), font=font)

        frame_file = frame_dir / f"frame_{i:05d}.png"
        img.save(frame_file, "PNG")
