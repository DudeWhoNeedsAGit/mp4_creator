import torch
import torchaudio
import json
import re
from pathlib import Path
from torchaudio.pipelines import MMS_FA as bundle

# --- GLOBAL MODEL CONFIG ---
# We load this once at the module level so it stays in VRAM during batch processing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Initializing MMS_FA on {device}...")

model = bundle.get_model().to(device).eval()
tokenizer = bundle.get_tokenizer()
aligner = bundle.get_aligner()

def get_lyrics_alignment(vocals_path: Path, lyrics_file: Path, words_per_line=4):
    """
    Integrates MMS_FA alignment with viral-style word chunking.
    """
    # 1. Load and Resample Audio
    waveform, sr = torchaudio.load(str(vocals_path))
    if waveform.shape[0] > 1: 
        waveform = waveform.mean(0, keepdim=True)
    if sr != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)

    # 2. Robust Text Cleaning (Matches your forced_alignment.py logic)
    raw_text = Path(lyrics_file).read_text(encoding="utf-8").strip()
    use_upper = 'A' in tokenizer.dictionary
    
    # Split text while preserving some original formatting if needed
    # but cleaning for the model dictionary
    raw_words = raw_text.split()
    clean_words = []
    words_to_render = []

    for w in raw_words:
        # Step 1: Prep for model dictionary (MMS usually wants lowercase or uppercase)
        target = w.upper() if use_upper else w.lower()
        # Step 2: Remove characters NOT in the model dictionary
        cleaned = re.sub(r'[^A-Z]' if use_upper else r'[^a-z]', '', target)
        
        if cleaned:
            clean_words.append(cleaned)
            # For "Viral" look, we use uppercase for the video output
            words_to_render.append(w.upper())

    # 3. Model Inference
    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
        tokens = tokenizer(clean_words)
        token_spans = aligner(emission[0], tokens)

    # 4. Map spans to Timestamps
    num_frames = emission.size(1)
    ratio = waveform.size(1) / num_frames / bundle.sample_rate
    
    aligned_words = []
    for i, spans in enumerate(token_spans):
        start_time = spans[0].start * ratio
        end_time = spans[-1].end * ratio
        aligned_words.append({
            "word": words_to_render[i],
            "start": round(start_time, 3),
            "end": round(end_time, 3)
        })

    # 5. Segment into Viral Chunks (e.g., 4 words per screen)
    segments = []
    for i in range(0, len(aligned_words), words_per_line):
        chunk = aligned_words[i : i + words_per_line]
        if chunk:
            segments.append({"words": chunk})
            
    return segments

def ass_from_json(json_path: Path, ass_path: Path, beat_file: Path = None, font="Arial Black", fontsize=44, resolution="854x480"):
    """
    Generates industry-standard viral subtitles with smooth karaoke highlights.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    def fmt_time(t):
        t = float(t)
        h, m, s = int(t//3600), int((t%3600)//60), int(t%60)
        cs = int(round((t - int(t)) * 100))
        if cs == 100: s += 1; cs = 0
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    # Style: &H0000E6FF is a vibrant Suno-style Yellow/Gold
    # MarginV: 80 pushes it up slightly from the very bottom for social media safety zones
    lines = [f"""[Script Info]
ScriptType: v4.00+
PlayResX: {resolution.split('x')[0]}
PlayResY: {resolution.split('x')[1]}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Viral,{font},{fontsize},&H00FFFFFF,&H0000E6FF,&H00000000,&H80000000,-1,0,0,0,100,100,1,0,1,4,1,2,30,30,80,1

[Events]
Format: Layer, Start, End, Style, Text
"""]

    for seg in segments:
        words = seg["words"]
        line_start = words[0]["start"]
        line_end = words[-1]["end"]
        
        payload = []
        for idx, w in enumerate(words):
            # Calculate fill duration for \kf
            duration = w["end"] - w["start"]
            
            # If the next word starts immediately, extend the highlight slightly for smoothness
            if idx < len(words) - 1:
                gap = words[idx+1]["start"] - w["end"]
                if gap < 0.2:
                    duration += gap

            dur_cs = max(1, int(round(duration * 100)))
            payload.append(f"{{\\kf{dur_cs}}}{w['word']} ")

        lines.append(f"Dialogue: 0,{fmt_time(line_start)},{fmt_time(line_end)},Viral,{''.join(payload)}")

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