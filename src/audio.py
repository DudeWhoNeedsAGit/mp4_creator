# src/audio.py

import subprocess
import librosa
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def normalize_audio(mp3_path):
    norm_wav_path = mp3_path.with_name(mp3_path.stem + "_norm.wav")
    if not norm_wav_path.exists():
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(mp3_path),
            "-af", "pan=mono|c0=.5*c0+.5*c1,volume=1.0",
            str(norm_wav_path)
        ], check=True)
    return norm_wav_path

def detect_beats(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    print(f"[Beats] Detected {len(beat_times)} beats, first 10: {beat_times[:10]}")
    return beat_times

def ensure_feynman_metadata(mp3_dir, force_flag):
    """
    Scans for audio files and ensures 'Feynman' is the Artist.
    """
    logger.info("🕵️ Checking metadata for 'Feynman' attribution...")
    audio_files = list(mp3_dir.glob("*.mp3")) + list(mp3_dir.glob("*.wav"))
    
    from .render_engine import get_audio_metadata

    for f in audio_files:
        meta = get_audio_metadata(str(f))
        artist = meta.get('artist', '<no artist tag>')
        
        if artist != "Feynman":
            if not force_flag:
                confirm = input(f"❓ File '{f.name}' artist is '{artist}'. Change to 'Feynman'? [y/N]: ")
                if confirm.lower() != 'y':
                    continue
            
            logger.info(f"📝 Updating {f.name} metadata to Feynman...")
            temp_f = f.with_suffix(".tmp" + f.suffix)
            cmd = [
                "ffmpeg", "-y", "-i", str(f),
                "-metadata", "artist=Feynman",
                "-codec", "copy", str(temp_f)
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            temp_f.replace(f)