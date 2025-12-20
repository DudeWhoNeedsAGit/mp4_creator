import subprocess
import json
import re
from pathlib import Path
from torchfa import TorchaudioForcedAligner
import torch
import numpy as np

def get_lyrics_alignment(vocals_path, lyrics_file, words_per_line=8):
    with open(lyrics_file, 'r', encoding="utf-8") as f:
        transcript = f.read().strip().replace('\n', ' ')
    
    aligner = TorchaudioForcedAligner(device='cuda' if torch.cuda.is_available() else 'cpu')
    cut = aligner.align_audios(str(vocals_path), transcript)
    
    # alignments is a list of NamedTuples (AlignmentItem)
    alignments = cut.supervisions[0].alignment['word']
    
    segments = []
    current_words = []
    
    for i, a in enumerate(alignments):
        # FIX: Use attribute access (dot notation) for NamedTuples
        word_text = a.symbol
        # torchfa/lhotse uses .start and .duration (start + duration = end)
        w_start = a.start
        w_end = a.start + a.duration
        
        current_words.append({'word': word_text, 'start': w_start, 'end': w_end})
        
        is_last = (i == len(alignments) - 1)
        
        # Check for punctuation to break lines
        has_punctuation = any(p in word_text for p in [".", "!", "?", ","])
        
        # Check for long silences
        long_pause = False
        if not is_last:
            next_a = alignments[i+1]
            long_pause = (next_a.start - w_end) > 1.0

        if len(current_words) >= words_per_line or has_punctuation or long_pause or is_last:
            segments.append({'words': current_words})
            current_words = []
            
    return segments

def ass_from_json(json_path: Path, ass_path: Path, beat_file: Path = None, font="DejaVu Sans", fontsize=60, resolution="854x480"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    beats = np.array([])
    if beat_file and beat_file.exists():
        with open(beat_file, "r", encoding="utf-8") as f:
            beats = np.array(json.load(f))

    if isinstance(data, list):
        segments = data
    elif "segments" in data:
        segments = data["segments"]
    else:
        segments = []

    def fmt_time_sec(t):
        t = float(t)
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        cs = int(round((t - int(t)) * 100))
        if cs == 100:
            s += 1
            cs = 0
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    lines = [f"""[Script Info]
ScriptType: v4.00+
PlayResX: {resolution.split('x')[0]}
PlayResY: {resolution.split('x')[1]}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Lyrics,{font},{fontsize},&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,3,2,2,50,50,60,1
Style: Highlight,{font},{fontsize},&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,3,2,2,50,50,60,1

[Events]
Format: Layer, Start, End, Style, Text
"""]

    for seg in segments:
        if "words" not in seg or not seg["words"]:
            continue
            
        words = seg["words"]
        start_line = float(words[0]["start"])
        end_line = float(words[-1]["end"])
        
        line_payload = []
        for idx, w in enumerate(words):
            w_start = float(w["start"])
            w_end = float(w["end"])
            w_text = w["word"].strip().replace("{","").replace("}","")

            # Calculate duration for \k tag
            if idx < len(words) - 1:
                next_start = float(words[idx+1]["start"])
                # Bridge small gaps between words for smoother highlighting
                duration = next_start - w_start if (next_start - w_end) < 0.8 else w_end - w_start
            else:
                duration = w_end - w_start
            
            dur_cs = max(1, int(round(duration * 100)))
            line_payload.append(f"{{\\k{dur_cs}}}{w_text} ")

        dialogue = f"Dialogue: 0,{fmt_time_sec(start_line)},{fmt_time_sec(end_line)},Lyrics,{''.join(line_payload)}"
        lines.append(dialogue)

    ass_path.parent.mkdir(exist_ok=True, parents=True)
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def get_audio_duration(path):
    output = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of",
        "default=noprint_wrappers=1:nokey=1", str(path)
    ])
    return float(output)