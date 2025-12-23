# main.py
import argparse
import logging
from pathlib import Path
from src import process_file, ensure_feynman_metadata

# Global Constants
CONFIG = {
    'MP3_DIR': Path("./mp3"),
    'OUT_DIR': Path("./out"),
    'MP4_DIR': Path("/mnt/c/tmp/wsl_out/"), # Windows Host path
    'VIDEO_RES': (854, 480),
    'FPS': 30
}

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default=None)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--steps', type=str, default=None)
    parser.add_argument('--upscale', action='store_true')
    parser.add_argument('--test-render', action='store_true')
    args = parser.parse_args()

    # Step setup
    requested_steps = (
        set(args.steps.split(',')) if args.steps
        else {'normalize', 'vocals', 'beats', 'transcription', 'subtitles', 'render'}
    )

    # Initialize Directories
    CONFIG['MP4_DIR'].mkdir(parents=True, exist_ok=True)
    CONFIG['OUT_DIR'].mkdir(parents=True, exist_ok=True)

    # 1. Cleansing Check
    ensure_feynman_metadata(CONFIG['MP3_DIR'], args.force)

    # 2. Batch Processing
    all_files = sorted(list(CONFIG['MP3_DIR'].glob("*.mp3")) + list(CONFIG['MP3_DIR'].glob("*.wav")))
    logger.info(f"🔍 Found {len(all_files)} files in {CONFIG['MP3_DIR']}")

    for f in all_files:
        try:
            process_file(f, args, requested_steps, CONFIG)
        except Exception as e:
            logger.error(f"💥 Critical Failure on {f.name}: {e}", exc_info=True)

if __name__ == "__main__":
    main()