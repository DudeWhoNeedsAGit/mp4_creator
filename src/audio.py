import subprocess
import librosa

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