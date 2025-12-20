import subprocess
import json
from pathlib import Path
from torchfa import TorchaudioForcedAligner
import torch

def get_lyrics_alignment(vocals_path, lyrics_file):
    with open(lyrics_file, 'r') as f:
        transcript = f.read().strip().replace('\n', ' ')
    aligner = TorchaudioForcedAligner(device='cuda' if torch.cuda.is_available() else 'cpu')
    cut = aligner.align_audios(str(vocals_path), transcript)
    alignments = cut.supervisions[0].alignment['word']  # list of dicts: {'symbol': word, 'begin': start_sec, 'end': end_sec, 'score': prob}
    segments = [{'words': [{'word': a['symbol'], 'start': a['begin'], 'end': a['end']} for a in alignments]}]
    return segments  # Mimic WhisperX format for ass_from_json

def ass_from_json(json_path: Path, ass_path: Path, font="DejaVu Sans", fontsize=60, resolution="854x480"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Detect segments: WhisperX vs Aeneas
    if isinstance(data, list):
        segments = data
    elif "segments" in data:  # WhisperX
        segments = data["segments"]
    elif "fragments" in data:  # Aeneas
        segments = data["fragments"]
    else:
        segments = []

    def fmt_time_sec(t):
        t = float(t)  # <-- add this
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        cs = int((t - int(t)) * 100)
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


    lines = [f"""[Script Info]
ScriptType: v4.00+
PlayResX: {resolution.split('x')[0]}
PlayResY: {resolution.split('x')[1]}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Lyrics,{font},{fontsize},&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,3,2,2,50,50,80,1
Style: Highlight,{font},{fontsize},&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,3,2,2,50,50,80,1

[Events]
Format: Layer, Start, End, Style, Text
"""]

    for seg in segments:
        if "words" in seg:  # WhisperX
            words = seg["words"]
            if not words:
                continue
            start_sec = words[0]["start"]
            end_sec = words[-1]["end"]
            line_text = []
            for i, w in enumerate(words):
                dur_cs = int((w["end"] - w["start"]) * 100)
                word_clean = w["word"].strip().replace("{","").replace("}","")
                style = "Highlight" if i == len(words)-1 else "Lyrics"
                line_text.append(f"{{\\k{dur_cs}}}{word_clean} ")
            dialogue = f"Dialogue: 0,{fmt_time_sec(start_sec)},{fmt_time_sec(end_sec)},Lyrics,{''.join(line_text)}"
            lines.append(dialogue)

        elif "begin" in seg and "end" in seg:  # Aeneas
            start_sec = seg["begin"]
            end_sec = seg["end"]
            text = " ".join(seg.get("lines", []))
            dialogue = f"Dialogue: 0,{fmt_time_sec(start_sec)},{fmt_time_sec(end_sec)},Lyrics,{text}"
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