import os
import glob
import argparse
from pathlib import Path
from upscaler import run_upscale  # This looks for your upscaler.py file

def batch_process(force=False):
    # 1. Setup Folders
    input_dir = Path("./output")
    output_dir = Path("./output/1080p_finished")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Identify Files
    all_videos = glob.glob(str(input_dir / "*.mp4"))
    
    # Filter out videos that are already in the finished folder
    to_process = []
    for vid in all_videos:
        file_name = os.path.basename(vid)
        # Skip if the file is already an "upscaled" version itself
        if "_1080p" in file_name:
            continue
            
        expected_output = output_dir / file_name.replace(".mp4", "_1080p.mp4")
        
        if expected_output.exists() and not force:
            print(f"⏭️  Skipping: {file_name} (Already upscaled)")
        else:
            to_process.append((vid, str(expected_output)))

    if not to_process:
        print("✅ No new videos to process. Use --force to re-run.")
        return

    print(f"🚀 Starting Batch: {len(to_process)} videos found.")

    # 3. Execution Loop
    for input_path, output_path in to_process:
        print(f"\n💎 Processing: {os.path.basename(input_path)}")
        try:
            run_upscale(input_path, output_path)
        except Exception as e:
            print(f"❌ Error upscaling {input_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Overwrite existing upscaled videos")
    args = parser.parse_args()
    
    batch_process(force=args.force)