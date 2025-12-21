Data flow: mp3 -> normalize_audio (mono WAV) -> Demucs (vocals WAV) -> WhisperX/transcription (JSON with timestamps) -> ass_from_json (ASS subtitles) -> FFmpeg render (overlay on video).

example commands:

./seup.sh

python main.py

./setup_aeneas.sh 


python align_lyrics_aeneas.py   --vocals "mp3/htdemucs/ANAMNESIS CORPUS_hbr_norm/vocals.wav"   --lyrics "mp3/ANAMNESIS CORPUS_hbr.txt"   --out "out/ANAMNESIS CORPUS_hbr/ANAMNESIS CORPUS_hbr_aeneas.json"

python main.py --force --use-custom-json "out/ANAMNESIS CORPUS_hbr/ANAMNESIS CORPUS_hbr_aeneas.json" --steps subtitles,render

Lyrics cleansing:
sed -i 's/\r$//' "./mp3/*.txt"  # normalize line endings
sed -i '/^\[.*\]/d; /^⸻/d' "./mp3/*.txt"

find ./mp3 -maxdepth 1 -name '*.txt' -exec sed -i 's/\r$//; /^\[.*\]/d; /^⸻/d' {} +


Change mp3 metadata with:
ffmpeg -i ./mp3/Esther.exe.mp3 -metadata title="Esther.exe" -metadata artist="Feynman" -codec copy Esther.exe.mp3

change only the author (needs sudo apt install id3v2)
for f in *.mp3; do id3v2 -A "NEW ARTIST" "$f"; done

ffmpeg -i ./mp3/"ABOMINATIO SYSTEMA _ IGNIS VERBI_norm.wav" -metadata artist="Feynman" -codec copy "temp.wav" && mv "temp.wav" ./mp3/"ABOMINATIO SYSTEMA _ IGNIS VERBI_norm.wav"

# change all wave files author:
for f in ./mp3/*.wav; do ffmpeg -i "$f" -metadata artist="Feynman" -codec copy "temp.wav" && mv "temp.wav" "$f"; done

# still image prompt:
A single still image heavy pulsing in intensity, like a soft breathing effect.
No camera movement, no zoom, no pan, no tilt.
No object movement, no deformation, no parallax.
The image remains perfectly static in position and shape.
Only a subtle rhythmic pulse in brightness and glow, sine-like easing.
Calm, minimal, seamless loop, designed for a music cover visualizer.
Cinematic lighting, stable framing, no flicker, no artifacts.