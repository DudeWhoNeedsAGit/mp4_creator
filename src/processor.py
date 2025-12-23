# src/processor.py
import json
import logging
import subprocess
from pathlib import Path
from .audio import normalize_audio
from .transcription import get_lyrics_alignment, ass_from_json
from .render_engine import run_integrated_render_gpu, get_audio_metadata
from upscaler import run_upscale 

logger = logging.getLogger(__name__)

def load_song_config(basename):
    config_path = Path("mp3_configs.json")
    if config_path.exists():
        with open(config_path, "r") as f:
            configs = json.load(f)
            return configs.get(basename.upper(), {})
    return {}

def process_file(mp3_path: Path, args, requested_steps, config):
    basename = mp3_path.stem
    final_output = Path(config['MP4_DIR']) / f"{basename}.mp4"
    
    if args.filename and args.filename != basename:
        return

    if final_output.exists() and not args.force:
        logger.info(f"⏭️  Skipping '{basename}': Output already exists.")
        return

    logger.info(f"🚀 Processing: {basename}")
    
    song_cfg = load_song_config(basename)
    do_upscale = args.upscale or song_cfg.get("upscale", False)
    theme = song_cfg.get("theme", "neon_puls")
    fps = config['FPS']
    video_res = config['VIDEO_RES']
    out_dir = config['OUT_DIR']
    mp3_dir = config['MP3_DIR']

    bg_source = next((f for f in [
        mp3_dir/f"{basename}_loop.mp4", 
        mp3_dir/f"{basename}_loopmr.mp4", 
        mp3_dir/f"{basename}.png", 
        mp3_dir/"bg.png"
    ] if f.exists()), None)

    whisper_out = out_dir / basename
    whisper_out.mkdir(parents=True, exist_ok=True)
    norm_path = mp3_path.with_name(f"{mp3_path.stem}_norm.wav")
    ass_file = whisper_out / f"{basename}.ass"
    json_file = whisper_out / "aligned_vocals.json"

    # --- Step 1-3: Prep ---
    if 'normalize' in requested_steps:
        normalize_audio(mp3_path)
    
    if 'vocals' in requested_steps: 
        subprocess.run(["demucs", "--two-stems=vocals", "-o", str(out_dir), str(norm_path)], check=True)
    
    if 'transcription' in requested_steps:
        model_name = "htdemucs" 
        vocals = out_dir / model_name / f"{basename}_norm" / "vocals.wav"
        if not vocals.exists(): vocals = out_dir / model_name / basename / "vocals.wav"
        segments = get_lyrics_alignment(vocals, mp3_path.with_suffix(".txt"))
        with open(json_file, "w") as f: 
            json.dump(segments, f, indent=2)

    # --- Step 4: Subtitles (Scaling Logic) ---
    if 'subtitles' in requested_steps:
        # IMPORTANT: Target the final resolution for the ASS script
        if do_upscale:
            target_res = (1920, 1080)
            base_fs = song_cfg.get("font_size", 20)
            # Scale factor (1080 / 480 = 2.25)
            fs = int(base_fs * 2.25) 
        else:
            target_res = (854, 480)
            fs = song_cfg.get("font_size", 14)

        logger.info(f"Generating Subtitles at {target_res[0]}p with FontSize {fs}")
        ass_from_json(json_file, ass_file, fontsize=fs, resolution=target_res)

    # --- Step 5: Render ---
    if 'render' in requested_steps:
        metadata = get_audio_metadata(str(norm_path))
        test_limit = (12 * fps) if args.test_render else None

        if do_upscale:
            temp_480 = out_dir / f"{basename}_raw_480.mp4"
            logger.info("🎬 Rendering 480p RAW Pass...")
            run_integrated_render_gpu(
                audio_path=norm_path, 
                ass_file=None, 
                bg_source=bg_source, 
                output_path=temp_480, 
                resolution=video_res, 
                fps=fps, 
                theme_name=theme,
                limit_frames=test_limit
            )
            
            logger.info("💎 AI Upscaling + 1080p Font Scaling...")
            # Pass the song_cfg to run_upscale so it knows the custom font sizes
            run_upscale(
                input_path=temp_480, 
                output_path=final_output, 
                ass_path=ass_file, 
                metadata=metadata,
                limit_frames=test_limit,
                base_font_size=song_cfg.get("font_size", 20)
            )
            if temp_480.exists(): temp_480.unlink()
        else:
            run_integrated_render_gpu(
                audio_path=norm_path, 
                ass_file=ass_file, 
                bg_source=bg_source, 
                output_path=final_output, 
                resolution=video_res, 
                fps=fps, 
                theme_name=theme,
                limit_frames=test_limit
            )
    
    logger.info(f"✅ Done: {basename}")