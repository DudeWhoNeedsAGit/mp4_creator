import torch
import torchaudio
import json
import re
from pathlib import Path
from tqdm import tqdm
from torchaudio.pipelines import MMS_FA as bundle

# --- GLOBAL CONFIG & OPTIMIZATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading and compiling model (this may take 1-2 mins)...")
# MMS_FA is the gold standard for forced alignment in torchaudio v2+
model = bundle.get_model().to(device).eval()
model = torch.compile(model) # Drastically speeds up inference after 1st run
tokenizer = bundle.get_tokenizer()
aligner = bundle.get_aligner()

def clean_text(text):
    """Removes Hebrew and special characters, leaving only A-Z and spaces."""
    # This ensures no KeyError is ever thrown by the tokenizer
    text = text.upper()
    return re.sub(r'[^A-Z\s]', '', text)

def debug_tokenizer(tokenizer, words):
    """Prints dictionary info and validates every character."""
    dict_keys = list(tokenizer.dictionary.keys())
    print(f"--- DEBUG: Model Dictionary ---")
    print(f"Dictionary Size: {len(dict_keys)}")
    print(f"Sample Keys: {dict_keys[:10]} ... {dict_keys[-10:]}")
    
    # Check if 'I' or 'i' exists
    print(f"Contains 'I': {'I' in tokenizer.dictionary}")
    print(f"Contains 'i': {'i' in tokenizer.dictionary}")
    
    for i, word in enumerate(words):
        for char in word:
            if char not in tokenizer.dictionary:
                print(f"CRITICAL: Character '{char}' in word '{word}' (index {i}) is NOT in dictionary!")

def align_lyrics(vocals_path: Path, lyrics_file: Path):
    # 1. Load Audio (same as before)
    waveform, sr = torchaudio.load(str(vocals_path))
    if waveform.shape[0] > 1: waveform = waveform.mean(0, keepdim=True)
    if sr != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)

    # 2. Robust Cleaning
    raw_text = Path(lyrics_file).read_text(encoding="utf-8").strip()
    
    # MMS models usually prefer LOWERCASE. Let's check the dictionary.
    use_upper = 'A' in tokenizer.dictionary
    print(f"Model prefers {'UPPERCASE' if use_upper else 'LOWERCASE'}")

    words = raw_text.split()
    clean_words = []
    original_words_kept = []

    for w in words:
        # Step 1: Remove Hebrew/Special chars
        # Step 2: Convert to model's preferred case
        target = w.upper() if use_upper else w.lower()
        cleaned = re.sub(r'[^A-Z]' if use_upper else r'[^a-z]', '', target)
        
        if cleaned:
            clean_words.append(cleaned)
            original_words_kept.append(w)

    # --- DEBUG TRIGGER ---
    debug_tokenizer(tokenizer, clean_words)

    # 3. Inference
    print("Running Inference...")
    with torch.inference_mode(), torch.amp.autocast(device_type="cuda"):
        emission, _ = model(waveform.to(device))
            
    # 4. Alignment
    print(f"Aligning {len(clean_words)} words...")
    tokens = tokenizer(clean_words)
    token_spans = aligner(emission[0], tokens)

    # 5. Build Timestamps
    # ratio = original samples / alignment frames / sample rate
    num_frames = emission.size(1)
    ratio = waveform.size(1) / num_frames / bundle.sample_rate
    
    seg_words = []
    for i, spans in enumerate(token_spans):
        # We map the alignment back to the original 'words' (possibly containing Hebrew)
        # but the timing is based on the 'clean_words'
        start_time = spans[0].start * ratio
        end_time = spans[-1].end * ratio
        seg_words.append({
            "word": words[i], # Keeps original text for your .ass file
            "start": round(start_time, 3),
            "end": round(end_time, 3)
        })
        
    return [{"words": seg_words}]

# --- Execution ---
vocals_path = Path("mp3/htdemucs/ANAMNESIS CORPUS_hbr_norm/vocals.wav")
lyrics_file = Path("mp3/ANAMNESIS CORPUS_hbr.txt")

print(f"Processing: {vocals_path.name}")
segments = align_lyrics(vocals_path, lyrics_file)

if segments:
    with open("aligned.json", "w", encoding="utf-8") as f:
        json.dump({"segments": segments}, f, ensure_ascii=False, indent=2)
    print("Done: aligned.json")