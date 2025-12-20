import argparse
import subprocess
from pathlib import Path
import json
import numpy as np
from src.audio import normalize_audio, detect_beats
from src.transcription import ass_from_json
from src.visualizer import generate_visualizer_frames
from src import get_lyrics_alignment

parser = argparse.ArgumentParser()
parser.add_argument('--force', action='store_true')
parser.add_argument('--use-custom-json', type=str, default=None)  # Path to custom vocals.json
parser.add_argument('--steps', type=str, default=None,
                    help="Comma-separated steps to run: normalize,vocals,beats,transcription,subtitles,visualizer,render")
args = parser.parse_args()

requested_steps = (
    set(args.steps.split(','))
    if args.steps
    else {'normalize', 'vocals', 'beats', 'transcription', 'subtitles', 'visualizer', 'render'}
)

MP3_DIR = Path("mp3")
OUT_DIR = Path("out")
MP4_DIR = Path("output")
VIDEO_RES = (854, 480)  # width x height
FONTSIZE = 40
FPS = 30

OUT_DIR.mkdir(exist_ok=True)
MP4_DIR.mkdir(exist_ok=True)

# -------------------
# Processing
# -------------------
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def normalize_step(mp3_path: Path, force: bool) -> Path:
    norm_path = mp3_path.with_name(mp3_path.stem + "_norm.wav")
    if norm_path.exists() and not force:
        logger.info(f"Using existing normalized audio: {norm_path}")
        return norm_path
    logger.info("Normalizing audio")
    return normalize_audio(mp3_path)

def vocals_step(norm_path: Path, force: bool) -> Path:
    vocals_dir = norm_path.parent / "htdemucs" / norm_path.stem
    vocals_path = vocals_dir / "vocals.wav"
    if vocals_path.exists() and not force:
        logger.info(f"Using existing vocals: {vocals_path}")
        return vocals_path
    logger.info("Separating vocals with Demucs")
    subprocess.run(["demucs", "--two-stems=vocals", "-o", str(norm_path.parent), str(norm_path)], check=True)
    if not vocals_path.exists():
        raise FileNotFoundError(f"Vocals not found: {vocals_path}")
    return vocals_path

def beats_step(norm_path: Path, whisper_out: Path, force: bool) -> np.ndarray:
    beat_file = whisper_out / "beat_times.json"
    if beat_file.exists() and not force:
        logger.info(f"Loading beats from {beat_file}")
        try:
            with open(beat_file) as f:
                data = f.read().strip()
                loaded = json.loads(data) if data else []
            return np.array(loaded)
        except json.JSONDecodeError:
            logger.warning(f"Corrupted {beat_file}, regenerating")
    logger.info("Detecting beats")
    beat_times = detect_beats(norm_path)
    with open(beat_file, 'w') as f:
        json.dump(beat_times.tolist(), f)
    logger.info(f"Saved beats to {beat_file}")
    return beat_times

# In transcription_step:
def transcription_step(vocals_path: Path, whisper_out: Path, force: bool, custom_json: str | None) -> Path:
    if custom_json:
        custom_path = Path(custom_json)
        if custom_path.exists():
            target_json = whisper_out / "custom_vocals.json"
            target_json.write_bytes(custom_path.read_bytes())
            logger.info(f"Using custom aligned JSON: {custom_path} → {target_json}")
            return target_json
        else:
            logger.warning(f"Custom JSON not found: {custom_json}")

    if any(whisper_out.glob("*.json")) and not force:
        logger.info("Using existing WhisperX output")
        json_files = list(whisper_out.glob("*.json"))
    else:
        logger.info("Running WhisperX transcription")
        cmd = [
            "whisperx", str(vocals_path),
            "--model", "large-v3",
            "--language", "en",
            "--device", "cuda",
            "--vad_onset", "0.1",
            "--vad_offset", "0.1",
            "--chunk_size", "5",
            "--compute_type", "float16",
            "--output_format", "json",
            "--output_dir", str(whisper_out)
        ]
        subprocess.run(cmd, check=True)
        json_files = list(whisper_out.glob("*.json"))

    if not json_files:
        raise FileNotFoundError("No JSON produced")
    json_file = json_files[0]
    logger.info(f"Using JSON: {json_file}")
    return json_file

def subtitles_step(json_file: Path, ass_file: Path, force: bool) -> None:
    if ass_file.exists() and not force:
        logger.info(f"Using existing ASS: {ass_file}")
        return
    logger.info("Generating ASS subtitles")
    ass_from_json(json_file, ass_file)

def visualizer_step(norm_path: Path, beat_times: np.ndarray, frame_dir: Path, force: bool) -> None:
    frame_dir.mkdir(exist_ok=True)
    sample_frame = frame_dir / "frame_00000.png"
    if sample_frame.exists() and not force:
        logger.info("Using existing frames")
        return
    logger.info("Generating visualizer frames")
    generate_visualizer_frames(norm_path, beat_times, frame_dir)

def render_step(frame_dir: Path, norm_path: Path, ass_file: Path, mp4_path: Path) -> None:
    logger.info("Rendering final MP4")
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", "mp3/bg.png",
        "-framerate", str(FPS), "-i", str(frame_dir / "frame_%05d.png"),
        "-i", str(norm_path),
        "-filter_complex",
        "[0:v]scale=854:480[bg];"
        "[bg][1:v]overlay=0:0[v];"
        f"[v]ass={ass_file}[va];"
        "[va]drawtext=fontfile=DejaVuSans.ttf:fontsize=30:fontcolor=white:bordercolor=black:borderw=2:"
        "x=10:y=10:text='%{pts\\:hms}'[outv]",
        "-map", "[outv]", "-map", "2:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest",
        str(mp4_path)
    ]
    subprocess.run(cmd, check=True)
    logger.info(f"Done: {mp4_path.name}")

def process_file(mp3_path: Path):
    basename = mp3_path.stem
    mp4_path = MP4_DIR / f"{basename}.mp4"

    logger.info(f"Processing {mp3_path.name}")

    whisper_out = OUT_DIR / basename
    whisper_out.mkdir(exist_ok=True)

    norm_path = mp3_path.with_name(mp3_path.stem + "_norm.wav")
    vocals_path = norm_path.parent / "htdemucs" / norm_path.stem / "vocals.wav"
    beat_file = whisper_out / "beat_times.json"
    ass_file = whisper_out / f"{basename}.ass"
    frame_dir = whisper_out / "frames"

    # --- normalize ---
    if 'normalize' in requested_steps:
        norm_path = normalize_step(mp3_path, args.force)
    elif not norm_path.exists():
        raise FileNotFoundError("Missing normalized audio")

    # --- vocals ---
    if 'vocals' in requested_steps:
        vocals_path = vocals_step(norm_path, args.force)
    elif not vocals_path.exists():
        raise FileNotFoundError("Missing vocals.wav in ", vocals_path)

    # --- beats ---
    if 'beats' in requested_steps:
        beat_times = beats_step(norm_path, whisper_out, args.force)
    else:
        beat_times = np.array(json.loads(beat_file.read_text()))

    # --- transcription ---
    if 'transcription' in requested_steps:
        json_file = transcription_step(
            vocals_path, whisper_out, args.force, args.use_custom_json
        )
    else:
        json_file = next(whisper_out.glob("*.json"))

    # --- subtitles ---
    if 'subtitles' in requested_steps:
        json_file = Path(args.use_custom_json) if args.use_custom_json else next(whisper_out.glob("*.json"))
        subtitles_step(json_file, ass_file, args.force)

    # --- visualizer ---
    if 'visualizer' in requested_steps:
        visualizer_step(norm_path, beat_times, frame_dir, args.force)
    elif not frame_dir.exists():
        raise FileNotFoundError("Missing frames")

    # --- render ---
    if 'render' in requested_steps:
        render_step(frame_dir, norm_path, ass_file, mp4_path)

def main():
    for mp3_file in MP3_DIR.glob("*.mp3"):
        process_file(mp3_file)

if __name__ == "__main__":
    main()
