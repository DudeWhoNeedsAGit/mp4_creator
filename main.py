import argparse
import subprocess
from pathlib import Path
import json
import numpy as np
import logging
from src.audio import normalize_audio, detect_beats
from src.transcription import ass_from_json
from src.visualizer import generate_visualizer_frames
from src import get_lyrics_alignment
from src.render_engine import run_integrated_render
# -------------------
# Configuration
# -------------------
parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default=None)
parser.add_argument('--force', action='store_true')
parser.add_argument('--use-custom-json', type=str, default=None)
parser.add_argument('--steps', type=str, default=None,
                    help="Comma-separated steps: normalize,vocals,beats,transcription,subtitles,visualizer,render")

args = parser.parse_args()

requested_steps = (
    set(args.steps.split(','))
    if args.steps
    else {'normalize', 'vocals', 'beats', 'transcription', 'subtitles', 'visualizer', 'render'}
)

MP3_DIR = Path("mp3")
OUT_DIR = Path("out")
MP4_DIR = Path("output")
VIDEO_RES = (854, 480)
FONTSIZE = 40
FPS = 30

OUT_DIR.mkdir(exist_ok=True)
MP4_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# -------------------
# Steps
# -------------------

def normalize_step(mp3_path: Path, force: bool) -> Path:
    norm_path = mp3_path.with_name(mp3_path.stem + "_norm.wav")
    if norm_path.exists() and not force:
        logger.info(f"Using existing normalized audio: {norm_path}")
        return norm_path
    logger.info(f"Normalizing audio: {mp3_path.name}")
    return normalize_audio(mp3_path)

def vocals_step(norm_path: Path, force: bool) -> Path:
    vocals_dir = norm_path.parent / "htdemucs" / norm_path.stem
    vocals_path = vocals_dir / "vocals.wav"
    if vocals_path.exists() and not force:
        logger.info(f"Using existing vocals: {vocals_path}")
        return vocals_path
    logger.info("Separating vocals with Demucs")
    subprocess.run(["demucs", "--two-stems=vocals", "-o", str(norm_path.parent), str(norm_path)], check=True)
    return vocals_path

def beats_step(norm_path: Path, whisper_out: Path, force: bool) -> np.ndarray:
    beat_file = whisper_out / "beat_times.json"
    if beat_file.exists() and not force:
        logger.info(f"Loading beats from {beat_file}")
        with open(beat_file) as f:
            return np.array(json.load(f))
    logger.info("Detecting beats")
    beat_times = detect_beats(norm_path)
    with open(beat_file, 'w') as f:
        json.dump(beat_times.tolist(), f)
    return beat_times

def transcription_step(vocals_path: Path, lyrics_path: Path, whisper_out: Path, force: bool) -> Path:
    target_json = whisper_out / "aligned_vocals.json"
    
    if target_json.exists() and not force:
        return target_json

    if not lyrics_path.exists():
        raise FileNotFoundError(f"Lyrics file not found for batch processing: {lyrics_path}")

    logger.info(f"Running Forced Alignment for: {lyrics_path.name}")
    segments = get_lyrics_alignment(vocals_path, lyrics_path)
    
    with open(target_json, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
        
    return target_json

def subtitles_step(json_file: Path, ass_file: Path, force: bool, beat_file: Path = None) -> None:
    if ass_file.exists() and not force:
        return
    logger.info(f"Generating ASS: {ass_file.name}")
    ass_from_json(json_file, ass_file, beat_file=beat_file)

def visualizer_step(norm_path: Path, beat_times: np.ndarray, frame_dir: Path, force: bool) -> None:
    frame_dir.mkdir(exist_ok=True)
    if list(frame_dir.glob("*.png")) and not force:
        return
    logger.info("Generating visualizer frames")
    generate_visualizer_frames(norm_path, beat_times, frame_dir)

def render_step(frame_dir: Path, norm_path: Path, ass_file: Path, mp4_path: Path, bg_source: Path) -> Path:
    logger.info(f"Final Render with background: {bg_source.name}")
    
    # Sanitize ASS path for FFmpeg filter
    ass_path_fixed = str(ass_file.absolute()).replace("\\", "/").replace(":", "\\:")
    
    # Determine if the background is a video or an image
    is_video = bg_source.suffix.lower() == ".mp4"
    
    # Build Input list
    # Input 0: Background (Video or Image)
    # Input 1: Visualizer Frames
    # Input 2: Audio
    cmd = ["ffmpeg", "-y"]
    
    if is_video:
        # Stream loop for video
        cmd += ["-stream_loop", "-1", "-i", str(bg_source)]
    else:
        # Standard loop for image
        cmd += ["-loop", "1", "-i", str(bg_source)]

    cmd += [
        "-framerate", str(FPS), "-i", str(frame_dir / "frame_%05d.png"),
        "-i", str(norm_path),
        "-filter_complex",
        # [0:v] is the background, [1:v] is the visualizer overlay
        f"[0:v]scale={VIDEO_RES[0]}:{VIDEO_RES[1]},setsar=1[bg];"
        f"[1:v]scale={VIDEO_RES[0]}:{VIDEO_RES[1]},setsar=1[vis];"
        f"[bg][vis]overlay=0:0[v];"
        f"[v]ass='{ass_path_fixed}'[outv]",
        "-map", "[outv]", 
        "-map", "2:a",
        "-c:v", "libx264", 
        "-pix_fmt", "yuv420p", 
        "-crf", "18",
        "-c:a", "aac", 
        "-b:a", "192k", 
        "-shortest",
        str(mp4_path)
    ]
    
    subprocess.run(cmd, check=True)

# -------------------
# Batch Processor
# -------------------

# -------------------
# Processing Logic
# -------------------

def process_file(mp3_path: Path):
    basename = mp3_path.stem
    mp4_path = MP4_DIR / f"{basename}.mp4"
    
    if args.filename is not None and args.filename != basename:
        logger.info(f"--- Specified filename: {args.filename}: skipping  {basename} ---")
        return

    elif mp4_path.exists() and not args.force:
        logger.info(f"--- Skipping {basename}: MP4 exists ---")
        return

    # --- Background Discovery Logic ---
    # Priority: 1. song_loop.mp4 | 2. song.png | 3. bg.png
    loop_video = MP3_DIR / f"{basename}_loop.mp4"
    song_image = MP3_DIR / f"{basename}.png"
    default_bg = MP3_DIR / "bg.png"

    if loop_video.exists():
        bg_source = loop_video
    elif song_image.exists():
        bg_source = song_image
    elif default_bg.exists():
        bg_source = default_bg
    else:
        raise FileNotFoundError(f"No background source found for {basename} in {MP3_DIR}")

    logger.info(f"--- bg_source {bg_source} ---")


    logger.info(f"--- Processing {basename} ---")

    whisper_out = OUT_DIR / basename
    whisper_out.mkdir(exist_ok=True)
    lyrics_path = mp3_path.with_suffix(".txt")
    norm_path = mp3_path.with_name(f"{basename}_norm.wav")

    vocals_path = mp3_path.parent / "htdemucs" / basename / "vocals.wav"
    beat_file = whisper_out / "beat_times.json"
    ass_file = whisper_out / f"{basename}.ass"
    frame_dir = whisper_out / "frames"

    logger.info(f"--- Starting Batch Job: {basename} ---")

    if 'normalize' in requested_steps:
        norm_path = normalize_step(mp3_path, args.force)
    
    if 'vocals' in requested_steps:
        vocals_path = vocals_step(norm_path, args.force)

    if 'beats' in requested_steps:
        beat_times = beats_step(norm_path, whisper_out, args.force)
    else:
        beat_times = np.array(json.loads(beat_file.read_text())) if beat_file.exists() else None

    if 'transcription' in requested_steps:
        json_file = transcription_step(vocals_path, lyrics_path, whisper_out, args.force)
    else:
        json_file = whisper_out / "aligned_vocals.json"

    if 'subtitles' in requested_steps:
        subtitles_step(json_file, ass_file, args.force, beat_file=beat_file)

    # if 'visualizer' in requested_steps:
    #     visualizer_step(norm_path, beat_times, frame_dir, args.force)

    # if 'render' in requested_steps:
    #     render_step(frame_dir, norm_path, ass_file, mp4_path, bg_source)
    if 'render' in requested_steps:
        run_integrated_render(
            audio_path=norm_path,
            ass_file=ass_file,
            bg_source=bg_source,
            output_path=mp4_path,
            resolution=VIDEO_RES,
            fps=FPS
        )

def main():
    # Only pick up .mp3 files (skips .wav or .txt)
    files = sorted(list(MP3_DIR.glob("*.mp3")))
    if not files:
        logger.error(f"No MP3 files found in {MP3_DIR}")
        return

    for mp3_file in files:
        try:
            process_file(mp3_file)
        except Exception as e:
            logger.error(f"Failed to process {mp3_file.name}: {e}")
            continue

if __name__ == "__main__":
    main()