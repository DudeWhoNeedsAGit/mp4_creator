# src/__init__.py

__all__ = [
    "process_file",
    "normalize_audio",
    "detect_beats",
    "ass_from_json",
    "generate_visualizer_frames",
    "get_lyrics_alignment",
    "render_engine",
]

from .audio import normalize_audio, detect_beats
from .transcription import ass_from_json
from .visualizer import generate_visualizer_frames
from .transcription import get_lyrics_alignment
from .render_engine import stream_visualizer_frames, run_integrated_render