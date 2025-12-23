import torch
import torchaudio
import json
import re
from pathlib import Path
from torchaudio.pipelines import MMS_FA as bundle
import logging
logger = logging.getLogger(__name__)

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

    logger.info(f"✨ PHASE 1: Transcription - Inference started ...")
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

def ass_from_json(json_path, ass_path, beat_file=None, font="Arial Black", fontsize=32, resolution="854x480", song_key=None, highlight=None):
    """
    Generates an ASS subtitle file with karaoke effects and dynamic scaling.
    Accepts an optional 'highlight' hex string to bypass internal config lookup.
    """
    import json
    import logging
    from pathlib import Path

    logger = logging.getLogger("subtitle_gen")
    
    def fmt_time(seconds):
        """Converts seconds to ASS time format H:MM:SS.cs"""
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h)}:{int(m):02d}:{s:05.2f}"

    # 1. Load the segments data
    json_p = Path(json_path)
    with open(json_p, "r", encoding="utf-8") as f:
        segments = json.load(f)

    # 2. Key Discovery & Config Loading
    # If highlight is provided from processor.py, we use it. 
    # Otherwise, we perform the internal lookup.
    highlight_hex = highlight if highlight else "#00E6FF"
    final_fontsize = fontsize
    config_path = Path("mp3_configs.json")

    if highlight is None and config_path.exists():
        raw_stem = json_p.stem
        search_origin = json_p.parent.name if raw_stem.lower() == "aligned_vocals" else raw_stem
        if song_key is None:
            song_key = search_origin.replace("_norm", "").replace("_norm".upper(), "").replace(" ", "_").replace("-", "_").upper()
        
        try:
            with open(config_path, "r") as f:
                all_configs = json.load(f)
                # Flexible lookup
                conf = all_configs.get(song_key, next((all_configs[k] for k in all_configs if k in song_key), {}))
                if conf:
                    highlight_hex = conf.get("highlight", highlight_hex)
                    final_fontsize = conf.get("font_size", final_fontsize)
                    logger.info(f"✅ Found internal highlight_hex: {highlight_hex}")
        except Exception as e:
            logger.error(f"Error reading {config_path}: {e}")
    else:
        if highlight:
            logger.info(f"🎨 Using injected highlight: {highlight}")

    # 4. Color Conversion (BGR for ASS)
    def hex_to_ass(h):
        h = h.lstrip('#')
        # Standard Hex is RGB, ASS is BGR
        r, g, b = h[0:2], h[2:4], h[4:6]
        return f"&H00{b}{g}{r}"

    ass_highlight = hex_to_ass(highlight_hex)
    
    # 5. Handle Resolution & Scaling Logic
    if isinstance(resolution, tuple):
        res_x, res_y = resolution
    else:
        res_x, res_y = map(int, resolution.split('x'))

    # DYNAMIC SCALING: Adjust for higher resolutions (e.g. 1080p)
    scale_factor = res_y / 480.0
    
    # Apply scaling to the fontsize if it hasn't been scaled by the caller
    if res_y > 720 and final_fontsize < 40:
        final_fontsize = int(final_fontsize * scale_factor)

    # Calculate positioning 
    center_y = res_y - int(100 * scale_factor) 
    line_spacing = int((final_fontsize + 15) * 1.1) 
    outline_thickness = max(1, int(2 * scale_factor))
    shadow_depth = max(1, int(1 * scale_factor))

    total_duration = segments[-1]["words"][-1]["end"] + 5.0 if segments else 300.0

    # 6. Build .ASS Header
    lines = [f"""[Script Info]
ScriptType: v4.00+
PlayResX: {res_x}
PlayResY: {res_y}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Current,{font},{final_fontsize},&H00FFFFFF,{ass_highlight},&H00000000,&H80000000,-1,0,0,0,100,100,1,0,1,{outline_thickness},{shadow_depth},2,30,30,80,1
Style: Faded,{font},{int(final_fontsize*0.8)},&H80AAAAAA,&H80AAAAAA,&H00000000,&H00000000,-1,0,0,0,100,100,1,0,1,1,0,2,30,30,80,1

[Events]
Format: Layer, Start, End, Style, Text
"""]

    # 7. Generate Dialogue Lines
    for i in range(len(segments)):
        current_words = segments[i]["words"]
        display_start = current_words[0]["start"]
        display_end = segments[i+1]["words"][0]["start"] if i < len(segments) - 1 else total_duration

        start_str = fmt_time(display_start)
        end_str = fmt_time(display_end)

        # Faded Past Text
        if i > 0:
            past_text = " ".join([w["word"] for w in segments[i-1]["words"]])
            lines.append(f"Dialogue: 1,{start_str},{end_str},Faded,{{\\fad(0,500)\\pos({res_x//2},{center_y - line_spacing})}}{past_text}")

        # Current Karaoke Text
        payload = []
        for idx, w in enumerate(current_words):
            duration = w["end"] - w["start"]
            # Join small gaps
            if idx < len(current_words) - 1:
                gap = current_words[idx+1]["start"] - w["end"]
                if 0 < gap < 0.2: duration += gap
            dur_cs = max(1, int(round(duration * 100)))
            payload.append(f"{{\\kf{dur_cs}}}{w['word']} ")
        
        lines.append(f"Dialogue: 2,{start_str},{end_str},Current,{{\\pos({res_x//2},{center_y})}}{''.join(payload)}")

        # Faded Future Text
        if i < len(segments) - 1:
            future_text = " ".join([w["word"] for w in segments[i+1]["words"]])
            lines.append(f"Dialogue: 1,{start_str},{end_str},Faded,{{\\fad(300,0)\\pos({res_x//2},{center_y + line_spacing})}}{future_text}")

    # 8. Save
    ass_path.parent.mkdir(exist_ok=True, parents=True)
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return True