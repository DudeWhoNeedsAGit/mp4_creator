# src/__init__.py
import subprocess
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from .audio import normalize_audio, detect_beats, ensure_feynman_metadata
from .transcription import ass_from_json, get_lyrics_alignment
from .visualizer import generate_visualizer_frames
from .render_engine import (
    stream_visualizer_frames, 
    run_integrated_render, 
    run_integrated_render_gpu,
    get_audio_metadata
)
from .processor import process_file, run_design_gallery

__all__ = [
    "process_file",
    "run_design_gallery",
    "normalize_audio",
    "detect_beats",
    "ensure_feynman_metadata",
    "ass_from_json",
    "get_lyrics_alignment",
    "generate_visualizer_frames",
    "run_integrated_render",
    "run_integrated_render_gpu",
    "get_audio_metadata",
]

def has_audio(file_path):
    """Checks if a video file contains an audio stream."""
    cmd = [
        "ffprobe", "-v", "error", "-show_streams", 
        "-select_streams", "a", "-print_format", "json", str(file_path)
    ]
    output = subprocess.check_output(cmd).decode()
    data = json.loads(output)
    return len(data.get("streams", [])) > 0

def strip_audio(file_path):
    """Removes audio from a video file in-place (via temp file)."""
    temp_path = file_path.with_name(f"tmp_{file_path.name}")
    logger.info(f"🔇 Stripping audio from {file_path.name}...")
    
    # -an removes audio, -c:v copy ensures NO re-encoding (instant speed)
    cmd = ["ffmpeg", "-y", "-i", str(file_path), "-an", "-c:v", "copy", str(temp_path)]
    subprocess.run(cmd, check=True, capture_output=True)
    
    file_path.unlink()  # Delete original
    temp_path.rename(file_path) # Replace with muted version