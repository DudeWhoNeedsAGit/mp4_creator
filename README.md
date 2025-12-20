Data flow: mp3 -> normalize_audio (mono WAV) -> Demucs (vocals WAV) -> WhisperX/transcription (JSON with timestamps) -> ass_from_json (ASS subtitles) -> FFmpeg render (overlay on video).

example commands:

./seup.sh

python main.py

./setup_aeneas.sh 


python align_lyrics_aeneas.py   --vocals "mp3/htdemucs/ANAMNESIS CORPUS_hbr_norm/vocals.wav"   --lyrics "mp3/ANAMNESIS CORPUS_hbr.txt"   --out "out/ANAMNESIS CORPUS_hbr/ANAMNESIS CORPUS_hbr_aeneas.json"

python main.py --force --use-custom-json "out/ANAMNESIS CORPUS_hbr/ANAMNESIS CORPUS_hbr_aeneas.json" --steps subtitles,render
