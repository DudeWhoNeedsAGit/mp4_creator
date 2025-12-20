import os
import subprocess
from pathlib import Path
import json
import librosa
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MP3_DIR = Path("mp3")
OUT_DIR = Path("out")
MP4_DIR = Path("output")
VIDEO_RES = (854, 480)  # width x height
FONT = "DejaVuSans.ttf"  # path to TTF font
FONTSIZE = 40
FPS = 30

OUT_DIR.mkdir(exist_ok=True)
MP4_DIR.mkdir(exist_ok=True)

# -------------------
# Utilities
# -------------------

def fmt_time_sec(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    cs = int((t - int(t)) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def normalize_audio(mp3_path):
    norm_path = mp3_path.with_name(mp3_path.stem + "_norm.wav")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(mp3_path),
        "-af", "pan=mono|c0=.5*c0+.5*c1,volume=1.0",
        str(norm_path)
    ], check=True)
    return norm_path

def detect_beats(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    print(f"[Beats] Detected {len(beat_times)} beats, first 10: {beat_times[:10]}")
    return beat_times

import json
from pathlib import Path

def ass_from_json(json_path: Path, ass_path: Path, font="DejaVu Sans", fontsize=60, resolution="854x480"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])

    def fmt_time_sec(t):
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        cs = int((t - int(t)) * 100)
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    lines = [f"""[Script Info]
ScriptType: v4.00+
PlayResX: {resolution.split('x')[0]}
PlayResY: {resolution.split('x')[1]}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Lyrics,{font},{fontsize},&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,3,2,2,50,50,80,1
Style: Highlight,{font},{fontsize},&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,3,2,2,50,50,80,1

[Events]
Format: Layer, Start, End, Style, Text
"""]

    for seg in segments:
        words = seg.get("words", [])
        if not words:
            continue
        start_sec = words[0]["start"]
        end_sec = words[-1]["end"]
        line_text = []
        for i, w in enumerate(words):
            dur_cs = int((w["end"] - w["start"]) * 100)
            word_clean = w["word"].strip().replace("{","").replace("}","")
            style = "Highlight" if i == len(words)-1 else "Lyrics"  # Simple highlight on current word
            line_text.append(f"{{\\k{dur_cs}}}{word_clean} ")
        dialogue = f"Dialogue: 0,{fmt_time_sec(start_sec)},{fmt_time_sec(end_sec)},Lyrics,{''.join(line_text)}"
        lines.append(dialogue)

    ass_path.parent.mkdir(exist_ok=True, parents=True)
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def get_audio_duration(path):
    output = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of",
        "default=noprint_wrappers=1:nokey=1", str(path)
    ])
    return float(output)

# -------------------
# Visualizer generation
# -------------------
from tqdm import tqdm
import colorsys
import random

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

# -------------------
# Processing
# -------------------

def process_file(mp3_path: Path):
    basename = mp3_path.stem
    mp4_path = MP4_DIR / f"{basename}.mp4"
    if mp4_path.exists():
        print(f"Skipping {mp3_path.name}, output exists.")
        return

    lyrics_file = mp3_path.with_suffix(".txt")
    if not lyrics_file.exists():
        print(f"[Warning] No lyrics.txt for {basename}, skipping.")
        return

    print(f"Processing {mp3_path.name}...")

    # Normalize full mix
    norm_path = normalize_audio(mp3_path)

    # Separate vocals with Demucs
    vocals_dir = norm_path.parent / "htdemucs" / norm_path.stem
    subprocess.run([
        "demucs", "--two-stems=vocals",
        "-o", str(norm_path.parent),
        str(norm_path)
    ], check=True)
    vocals_path = vocals_dir / "vocals.wav"
    if not vocals_path.exists():
        raise FileNotFoundError(f"Vocals not found: {vocals_path}")

    # Detect beats on full mix
    beat_times = detect_beats(norm_path)

    # WhisperX on isolated vocals + force align with lyrics
    whisper_out = OUT_DIR / basename
    whisper_out.mkdir(exist_ok=True)
    cmd = [
        "whisperx",
        str(vocals_path),
        "--model", "large-v3",
        "--language", "en",
        "--device", "cuda",
        "--vad_onset", "0.1",
        "--vad_offset", "0.1",
        "--chunk_size", "5",
        "--compute_type", "float16",
        "--output_format", "json",
        # "--align", str(lyrics_file),  # force alignment (new flag name)
        "--output_dir", str(whisper_out)
    ]
    subprocess.run(cmd, check=True)

    # Find JSON
    json_files = list(whisper_out.glob("*.json"))
    if not json_files:
        print(f"[Warning] No JSON produced for {basename}, skipping ASS and MP4")
        return
    json_file = json_files[0]

    # Generate ASS
    ass_file = whisper_out / f"{basename}.ass"
    ass_from_json(json_file, ass_file)

    # Visualizer frames on full mix
    frame_dir = whisper_out / "frames"
    frame_dir.mkdir(exist_ok=True)
    generate_visualizer_frames(norm_path, beat_times, frame_dir)

    # Render MP4
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", str(frame_dir / "frame_%05d.png"),
        "-i", str(norm_path),
        "-vf", f"ass={ass_file}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest",
        str(mp4_path)
    ]
    subprocess.run(cmd, check=True)
    print(f"Done: {mp4_path.name}")


def main():
    for mp3_file in MP3_DIR.glob("*.mp3"):
        process_file(mp3_file)

if __name__ == "__main__":
    main()
