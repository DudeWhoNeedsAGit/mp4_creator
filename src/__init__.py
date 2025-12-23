# src/__init__.py

from .audio import normalize_audio, detect_beats, ensure_feynman_metadata
from .transcription import ass_from_json, get_lyrics_alignment
from .visualizer import generate_visualizer_frames
from .render_engine import (
    stream_visualizer_frames, 
    run_integrated_render, 
    run_integrated_render_gpu,
    get_audio_metadata
)
from .processor import process_file

__all__ = [
    "process_file",
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