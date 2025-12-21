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

def get_lyrics_alignment(vocals_path: Path, lyrics_file: Path, max_words_per_line=6, pause_threshold=0.8):
    """
    Improved Alignment: Groups words based on audio pauses and punctuation 
    instead of fixed word counts.
    """
    import torch
    import torchaudio
    import re
    from pathlib import Path

    # 1. Load and Resample Audio
    waveform, sr = torchaudio.load(str(vocals_path))
    if waveform.shape[0] > 1: 
        waveform = waveform.mean(0, keepdim=True)
    if sr != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)

    # 2. Text Prep: Keep raw words to check for punctuation/casing
    raw_text = Path(lyrics_file).read_text(encoding="utf-8").strip()
    raw_words = raw_text.split()
    
    use_upper = 'A' in tokenizer.dictionary
    clean_words = []
    words_to_render = []

    for w in raw_words:
        target = w.upper() if use_upper else w.lower()
        cleaned = re.sub(r'[^A-Z]' if use_upper else r'[^a-z]', '', target)
        if cleaned:
            clean_words.append(cleaned)
            # Use original punctuation for the video, but uppercase for "Viral" look
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

    # 5. Smart Segmentation (Feature: Pause & Punctuation Awareness)
    segments = []
    current_chunk = []

    for i in range(len(aligned_words)):
        word_data = aligned_words[i]
        current_chunk.append(word_data)
        
        # Determine if we should break here
        should_break = False
        
        # Case A: It's the last word
        if i == len(aligned_words) - 1:
            should_break = True
        else:
            next_word = aligned_words[i+1]
            gap = next_word["start"] - word_data["end"]
            
            # Case B: Significant pause (breath) detected
            if gap > pause_threshold:
                should_break = True
            
            # Case C: Punctuation detected (End of sentence)
            if any(char in word_data["word"] for char in [".", "!", "?"]):
                should_break = True
            
            # Case D: Safety Max (Don't let lines get too long for the screen)
            if len(current_chunk) >= max_words_per_line:
                should_break = True

        if should_break:
            segments.append({"words": current_chunk})
            current_chunk = []
            
    return segments

def ass_from_json(json_path, ass_path, beat_file=None, font="Arial Black", fontsize=32, resolution="854x480"):
    """
    Generates persistent 3-line viral subtitles.
    Fetches song-specific colors from mp3_configs.json based on filename.
    """
    import json
    from pathlib import Path

    # 1. Config Loading Logic
    config_path = Path("mp3_configs.json")
    song_key = Path(json_path).stem.replace(" ", "_").upper()
    highlight_color = "#00E6FF" # Default Gold
    
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                all_configs = json.load(f)
                # Check if this specific song has a config
                if song_key in all_configs:
                    highlight_color = all_configs[song_key].get("highlight", highlight_color)
                    fontsize = all_configs[song_key].get("font_size", fontsize)
                    print(f"[INFO] Applied config for {song_key}: Color={highlight_color}")
        except Exception as e:
            print(f"[WARNING] Could not read config file: {e}")

    # 2. Hex to ASS Conversion
    def hex_to_ass(hex_str):
        clean_hex = hex_str.lstrip('#')
        if len(clean_hex) == 6:
            r, g, b = clean_hex[0:2], clean_hex[2:4], clean_hex[4:6]
            return f"&H00{b}{g}{r}" 
        return "&H0000E6FF" 

    ass_highlight = hex_to_ass(highlight_color)

    # 3. Process JSON Data
    with open(json_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    def fmt_time(t):
        t = max(0, float(t))
        h, m, s = int(t//3600), int((t%3600)//60), int(t%60)
        cs = int(round((t - int(t)) * 100))
        if cs == 100: s += 1; cs = 0
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    res_x, res_y = map(int, resolution.split('x'))
    center_y = res_y - 100 
    line_spacing = fontsize + 15 

    total_duration = segments[-1]["words"][-1]["end"] + 5.0 if segments else 300.0

    lines = [f"""[Script Info]
ScriptType: v4.00+
PlayResX: {res_x}
PlayResY: {res_y}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Current,{font},{fontsize},&H00FFFFFF,{ass_highlight},&H00000000,&H80000000,-1,0,0,0,100,100,1,0,1,2,1,2,30,30,80,1
Style: Faded,{font},{int(fontsize*0.8)},&H80AAAAAA,&H80AAAAAA,&H00000000,&H00000000,-1,0,0,0,100,100,1,0,1,1,0,2,30,30,80,1

[Events]
Format: Layer, Start, End, Style, Text
"""]

    for i in range(len(segments)):
        current_words = segments[i]["words"]
        display_start = current_words[0]["start"]
        display_end = segments[i+1]["words"][0]["start"] if i < len(segments) - 1 else total_duration

        start_str = fmt_time(display_start)
        end_str = fmt_time(display_end)

        if i > 0:
            past_text = " ".join([w["word"] for w in segments[i-1]["words"]])
            lines.append(f"Dialogue: 1,{start_str},{end_str},Faded,{{\\fad(0,500)\\pos({res_x//2},{center_y - line_spacing})}}{past_text}")

        payload = []
        for idx, w in enumerate(current_words):
            duration = w["end"] - w["start"]
            if idx < len(current_words) - 1:
                gap = current_words[idx+1]["start"] - w["end"]
                if 0 < gap < 0.2: duration += gap
            dur_cs = max(1, int(round(duration * 100)))
            payload.append(f"{{\\kf{dur_cs}}}{w['word']} ")
        
        lines.append(f"Dialogue: 2,{start_str},{end_str},Current,{{\\pos({res_x//2},{center_y})}}{''.join(payload)}")

        if i < len(segments) - 1:
            future_text = " ".join([w["word"] for w in segments[i+1]["words"]])
            lines.append(f"Dialogue: 1,{start_str},{end_str},Faded,{{\\fad(300,0)\\pos({res_x//2},{center_y + line_spacing})}}{future_text}")

    ass_path.parent.mkdir(exist_ok=True, parents=True)
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))