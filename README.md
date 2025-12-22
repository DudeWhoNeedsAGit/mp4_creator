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

# installation fixes:
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# This fixes a known import error in the basicsr library
sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' $(pip show basicsr | grep Location | awk '{print $2}')/basicsr/data/degradations.py

# How to verify your 4070TI Super is ready:
python -c "import torch; print(f'GPU Found: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'ERROR: GPU NOT FOUND')"