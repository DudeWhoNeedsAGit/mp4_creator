This `README.md` is designed to serve as the definitive manual for your music visualizer project. It incorporates your specific data flow, installation fixes for the **4070TI Super**, and your custom shell logic for metadata and lyrics cleansing.

---

# 🎼 Feynman Music Visualizer & AI Upscaler

A high-performance automated pipeline for creating cinematic music videos. This tool transforms raw audio and text into synchronized, high-definition (1080p) visualizers with beat-responsive graphics and AI-upscaled backgrounds.

## 🛠 Project Data Flow

The system operates as a linear pipeline:

1. **Audio Prep:** `mp3` → `normalize_audio` (Mono WAV).
2. **Stems:** `Demucs` extracts `vocals.wav`.
3. **Alignment:** `WhisperX/Aeneas` → `JSON` (Word-level timestamps).
4. **Style:** `ass_from_json` generates high-res `.ass` subtitles.
5. **Render:** `FFmpeg` overlays Numba-rendered visualizers onto backgrounds.
6. **AI Upscale:** `Real-ESRGAN` scales 480p to 1080p with crisp UI/Text overlays.

---

## 🚀 Installation & Fixes

### 1. Basic Setup

```bash
./setup.sh
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

```

### 2. Fix `BasicsSR` Torchvision Error

Real-ESRGAN depends on `basicsr`, which has a known import bug in modern PyTorch. Run this to patch it:

```bash
sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' $(pip show basicsr | grep Location | awk '{print $2}')/basicsr/data/degradations.py

```

### 3. Verify Hardware (RTX 4070TI Super)

Ensure your CUDA environment is correctly mapped to your GPU:

```bash
python -c "import torch; print(f'GPU Found: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'ERROR: GPU NOT FOUND')"

```

---

## 🎮 Execution Commands

### Standard Automatic Run

Processes all files in `./mp3` and outputs to your Windows Host `/mnt/c/tmp/wsl_out/`.

```bash
python main.py

```

### Advanced Lyric Alignment (Aeneas)

Use this if the standard WhisperX alignment is struggling with complex vocals:

```bash
./setup_aeneas.sh 

python align_lyrics_aeneas.py \
  --vocals "mp3/htdemucs/SONG_NAME_norm/vocals.wav" \
  --lyrics "mp3/SONG_NAME.txt" \
  --out "out/SONG_NAME/SONG_NAME_aeneas.json"

# Re-run render using the custom alignment
python main.py --force --use-custom-json "out/SONG_NAME/SONG_NAME_aeneas.json" --steps subtitles,render

```

---

## 🧹 Maintenance & Cleansing Scripts

### Lyrics Pre-processing

Clean your `.txt` files to remove bracketed notes `[Chorus]` or decorative separators `⸻` and fix Windows line endings:

```bash
# Normalize line endings and remove tags/decorations for all files
find ./mp3 -maxdepth 1 -name '*.txt' -exec sed -i 's/\r$//; /^\[.*\]/d; /^⸻/d' {} +

```

### Metadata Correction (The "Feynman" Check)

Ensure all files are attributed to **Feynman** to avoid pipeline metadata errors.

**Batch Update for WAV files:**

```bash
for f in ./mp3/*.wav; do ffmpeg -i "$f" -metadata artist="Feynman" -codec copy "temp.wav" && mv "temp.wav" "$f"; done

```

**Specific MP3 Update:**

```bash
ffmpeg -i ./mp3/Song.mp3 -metadata title="Song Title" -metadata artist="Feynman" -codec copy Song_Fixed.mp3

```

**Alternative (id3v2):**

```bash
# Useful for mass-tagging if ffmpeg headers are stubborn
for f in *.mp3; do id3v2 -A "Feynman" "$f"; done

```

---

## 🖼 Background Generation Guide

If using AI (Midjourney/Runway) to generate loop backgrounds, use the following prompt structure for the best visualizer results:

> **Prompt:** A single still image heavy pulsing in intensity, like a soft breathing effect. No camera movement, no zoom, no pan, no tilt. No object movement, no deformation, no parallax. The image remains perfectly static in position and shape. Only a subtle rhythmic pulse in brightness and glow, sine-like easing. Calm, minimal, seamless loop, designed for a music cover visualizer. Cinematic lighting, stable framing, no flicker, no artifacts.

---

## 📂 Directory Configuration

* **`./mp3`**: Place your source audio and `.txt` lyrics here.
* **`./out`**: Internal workspace for stems and JSON.
* **`/mnt/c/tmp/wsl_out/`**: Final `.mp4` files are delivered here for Windows access.