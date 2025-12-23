# main.py
import argparse
import logging
import json
from pathlib import Path
from src import process_file, ensure_feynman_metadata, run_design_gallery
from src import has_audio, strip_audio # or wherever you put it

CONFIG = {
    'MP3_DIR': Path("./mp3"),
    'OUT_DIR': Path("./out"),
    'MP4_DIR': Path("/mnt/c/tmp/wsl_out/"),
    'VIDEO_RES': (854, 480),
    'FPS': 30
}

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    # 1. REMOVED required=True so the script can run in Batch Mode
    parser.add_argument('--filename', type=str, help="Specific file to process")
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--upscale', action='store_true')
    parser.add_argument('--test-render', action='store_true')
    args = parser.parse_args()

    # Setup
    CONFIG['MP4_DIR'].mkdir(parents=True, exist_ok=True)
    CONFIG['OUT_DIR'].mkdir(parents=True, exist_ok=True)
    ensure_feynman_metadata(CONFIG['MP3_DIR'], args.force)

    # --- NEW: VISUAL SANITIZATION ---
    logger.info("🧼 Checking background visuals for audio...")
    visual_files = list(CONFIG['MP3_DIR'].glob("*.mp4"))
    for v in visual_files:
        if has_audio(v):
            strip_audio(v)
            logger.info(f"🧼 Stripped audio from {v}")


    # 2. FILE DISCOVERY LOGIC
    targets = []
    if args.filename:
        # Single file mode
        path = CONFIG['MP3_DIR'] / f"{args.filename}.mp3"
        if not path.exists(): path = path.with_suffix(".wav")
        if path.exists(): targets.append(path)
    else:
        # Batch mode: Find all mp3/wav files
        logger.info("🔍 Scanning mp3 folder for new files...")
        all_files = list(CONFIG['MP3_DIR'].glob("*.mp3")) + list(CONFIG['MP3_DIR'].glob("*.wav"))
        
        for f in all_files:
            # Skip the helper normalized files
            if "_norm" in f.name: continue
            
            # 3. CHECK FOR EXISTING OUTPUT
            # We check for the _480p or _1080p suffix based on current mode
            res_tag = "_1080p" if args.upscale else "_480p"
            expected_out = CONFIG['MP4_DIR'] / f"{f.stem}{res_tag}.mp4"
            
            if not expected_out.exists() or args.force:
                targets.append(f)
            else:
                logger.info(f"⏭️  Skipping {f.name}: {res_tag} version already exists.")

    if not targets:
        logger.info("✅ All files are up to date. Nothing to process.")
        return

    # 4. EXECUTION
    for mp3_path in targets:
        logger.info(f"📂 Found new target: {mp3_path.name}")
        if args.test_render:
            run_design_gallery(mp3_path, args, CONFIG)
        else:
            requested_steps = {'normalize', 'vocals', 'transcription', 'subtitles', 'render'}
            process_file(mp3_path, args, requested_steps, CONFIG)

if __name__ == "__main__":
    main()