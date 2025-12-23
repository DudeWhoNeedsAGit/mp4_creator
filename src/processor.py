# src/processor.py
import json
import logging
import subprocess
from pathlib import Path
from .audio import normalize_audio
from .transcription import get_lyrics_alignment, ass_from_json
from .render_engine import run_integrated_render_gpu, get_audio_metadata, run_fast_ui_overlay, run_integrated_render
from upscaler import run_upscale 

logger = logging.getLogger(__name__)

def load_song_config(config_key):
    """Loads a specific entry with a whitespace/underscore insensitive search."""
    config_path = Path("mp3_configs.json")
    if not config_path.exists():
        return {}

    with open(config_path, "r") as f:
        configs = json.load(f)

    # Standardize function: remove all separators and make uppercase
    def standardize(s):
        return s.upper().replace("_", "").replace(" ", "")

    search_target = standardize(config_key)

    # 1. Check Song Specifics first (Root level)
    for key, value in configs.items():
        if key == "DESIGN_STYLES": continue
        if standardize(key) == search_target:
            logger.info(f"🎯 Exact Config Match: {key}")
            return value

    # 2. Check for Partial Match (If filename has extra tags like '_GOTH')
    for key, value in configs.items():
        if key == "DESIGN_STYLES": continue
        if standardize(key) in search_target:
            logger.info(f"🔍 Partial Config Match: {key}")
            return value

    # 3. Fallback to DESIGN_STYLES
    styles = configs.get("DESIGN_STYLES", {})
    for key, value in styles.items():
        if standardize(key) == search_target:
            return value

    logger.warning(f"⚠️ No config match for '{config_key}'. Using Toxic Green defaults.")
    return {}

def process_file(mp3_path: Path, args, requested_steps, config, override_config_key=None, font_mult=1.0):
    basename = mp3_path.stem
    fps = config['FPS']
    out_dir = config['OUT_DIR']
    mp3_dir = config['MP3_DIR']
    
    # --- 1. Config & Naming Logic ---
    cfg_key = override_config_key if override_config_key else basename
    song_cfg = load_song_config(cfg_key)
    theme = song_cfg.get("theme", "neon_puls")

    do_upscale = args.upscale #or song_cfg.get("upscale", False)
    res_suffix = "_1080p" if do_upscale else "_480p" # NEW: Resolution tag

    # Final Output naming: Song_CONFIG_x2.0_1080p.mp4 or Song_480p.mp4
    test_tag = f"_{cfg_key}_x{font_mult}" if args.test_render else ""
    final_output = Path(config['MP4_DIR']) / f"{basename}{test_tag}{res_suffix}.mp4"
    
    # The Clean Master is tied to the THEME. If theme is same, reuse the video.
    master_tag = "_test" if args.test_render else ""
    clean_master = out_dir / f"{basename}_{theme}_1080p_clean{master_tag}.mp4"

    if final_output.exists() and not args.force and not args.test_render:
        logger.info(f"⏭️  Skipping '{basename}': Output already exists.")
        return

    logger.info(f"🚀 Processing: {basename} (Config: {cfg_key}, FontMult: {font_mult})")

    # Background Discovery
    bg_source = next((f for f in [
        mp3_dir/f"{basename}_loop.mp4", 
        mp3_dir/f"{basename}_loopmr.mp4", 
        mp3_dir/f"{basename}.png", 
        mp3_dir/"bg.png"
    ] if f.exists()), None)

    whisper_out = out_dir / basename
    whisper_out.mkdir(parents=True, exist_ok=True)
    norm_path = mp3_path.with_name(f"{mp3_path.stem}_norm.wav")
    
    # Unique ASS file for this specific variation
    ass_file = whisper_out / f"{basename}{test_tag}.ass"
    json_file = whisper_out / "aligned_vocals.json"

    # --- Step 1-3: Prep (Standard Logic) ---
    if 'normalize' in requested_steps:
        logger.info(f"✨ PHASE 1: Normalization")
        normalize_audio(mp3_path)
    
    if 'vocals' in requested_steps: 
        logger.info(f"✨ PHASE 1: Vocals")
        subprocess.run(["demucs", "--two-stems=vocals", "-o", str(out_dir), str(norm_path)], check=True)
    
    if 'transcription' in requested_steps:
        logger.info(f"✨ PHASE 1: Transcription")
        # Avoid re-transcribing if JSON exists to speed up batch tests
        if not json_file.exists() or args.force:
            model_name = "htdemucs" 
            vocals = out_dir / model_name / f"{basename}_norm" / "vocals.wav"
            if not vocals.exists(): vocals = out_dir / model_name / basename / "vocals.wav"
            segments = get_lyrics_alignment(vocals, mp3_path.with_suffix(".txt"))
            with open(json_file, "w") as f: 
                json.dump(segments, f, indent=2)

    # --- Step 4: Subtitles (Scale-Aware Logic) ---
    if 'subtitles' in requested_steps:
        if do_upscale:
            target_res = (1920, 1080)
            base_fs = song_cfg.get("font_size", 20)
            # Scale for 1080p (2.25) AND the user requested multiplier (1.5 or 2.0)
            fs = int(base_fs * 2.25 * font_mult)
        else:
            target_res = (854, 480)
            fs = int(song_cfg.get("font_size", 14) * font_mult)

        logger.info(f"Creating ASS: Res={target_res}, FontSize={fs}")
        ass_from_json(json_file, ass_file, fontsize=fs, resolution=target_res, highlight=song_cfg.get("highlight"))

    # --- Step 5: Render (Master/Overlay Architecture) ---
    if 'render' in requested_steps:
        metadata = get_audio_metadata(str(norm_path))
        test_limit = (12 * fps) if args.test_render else None

        if do_upscale:
            # A. THE SLOW PASS: Check for/Create Clean 1080p Master
            if not clean_master.exists() or args.force:
                temp_480_raw = out_dir / f"tmp_raw_480_{theme}.mp4"
                logger.info(f"🎬 Creating Master ({theme})...")
                
                # Render 480p with NO text/subs
                run_integrated_render_gpu(
                    audio_path=norm_path, 
                    ass_file=None, 
                    bg_source=bg_source, 
                    output_path=temp_480_raw, 
                    resolution=config['VIDEO_RES'], 
                    fps=fps, 
                    theme_name=theme,
                    limit_frames=test_limit
                )
                
                # AI Upscale to 1080p (No metadata/ass here to keep it clean)
                run_upscale(
                    input_path=temp_480_raw, 
                    output_path=clean_master, 
                    ass_path=None, 
                    metadata=None,
                    limit_frames=test_limit
                )
                if temp_480_raw.exists(): temp_480_raw.unlink()
            else:
                logger.info(f"✨ Reusing Clean Master: {clean_master.name}")

            # B. THE FAST PASS: Burn in UI and Subtitles
            logger.info(f"🎨 Fast-Burning UI: {cfg_key}")
            run_fast_ui_overlay(
                clean_master_path=clean_master,
                output_path=final_output,
                ass_path=ass_file,
                metadata=metadata,
                base_font_size=song_cfg.get("font_size", 20) * font_mult
            )
        else:
            # Classic 480p rendering logic (One-pass)
            run_integrated_render_gpu(
                audio_path=norm_path, 
                ass_file=ass_file, 
                bg_source=bg_source, 
                output_path=final_output, 
                resolution=config['VIDEO_RES'], 
                fps=fps, 
                theme_name=theme,
                limit_frames=test_limit
            )
    
    logger.info(f"✅ Finished Variation: {cfg_key} (x{font_mult})")

def run_design_gallery(mp3_path, args, config):
    """
    Executes one slow master render (12s test), then loops through specific 
    DESIGN_STYLES using fast FFmpeg overlays.
    """
    import json
    import logging
    from pathlib import Path
    
    logger = logging.getLogger("gallery")
    basename = mp3_path.stem
    out_dir = config['OUT_DIR']
    
    # --- PHASE 1: THE HEAVY LIFTING ---
    logger.info(f"🎬 [PHASE 1] Creating 12s Clean Master for {basename}...")
    # Generate assets and the clean 1080p master video
    process_file(mp3_path, args, {'normalize', 'vocals', 'transcription', 'subtitles', 'render'}, config)

    # --- PHASE 2: CONFIG & ASSET PREP ---
    config_path = Path("mp3_configs.json")
    if not config_path.exists():
        logger.error("❌ mp3_configs.json not found. Cannot run gallery.")
        return

    with open(config_path, "r") as f:
        full_data = json.load(f)
    
    # Only iterate through keys inside DESIGN_STYLES to ignore song-specific settings
    all_styles = full_data.get("DESIGN_STYLES", {})
    if not all_styles:
        logger.warning("⚠️ 'DESIGN_STYLES' key not found in config. Using root keys.")
        all_styles = full_data

    # Resolve paths generated by Phase 1
    song_cfg = load_song_config(basename)
    theme = song_cfg.get("theme", "neon_puls")
    
    # Gallery always uses the '_test' master (12 seconds)
    clean_master = out_dir / f"{basename}_{theme}_1080p_clean_test.mp4"
    json_file = out_dir / basename / "aligned_vocals.json"
    norm_wav = mp3_path.with_name(f"{basename}_norm.wav")
    metadata = get_audio_metadata(str(norm_wav))

    if not clean_master.exists():
        logger.error(f"❌ Clean Master not found at {clean_master}. Phase 1 failed?")
        return

    # --- PHASE 3: THE FAST DESIGN LOOP ---
    logger.info(f"🎨 [PHASE 2] Generating {len(all_styles)*2} Gallery Variations...")

    for style_key, style_cfg in all_styles.items():
        # Safety: skip song entries that might have been left in root
        if not isinstance(style_cfg, dict): continue
        
        for mult in [1.0, 1.5]:
            # Naming includes resolution to prevent overwriting
            variation_name = f"{style_key}_x{mult}"
            variation_ass = out_dir / basename / f"{variation_name}.ass"
            output_mp4 = Path(config['MP4_DIR']) / f"{basename}_{variation_name}_1080p.mp4"

            logger.info(f"🖌️  Applying Style: {style_key} | Scale: {mult}x")

            # 1. Generate Subtitles
            # Extract base font size from style, default to 20
            base_fs = style_cfg.get("font_size", 20)
            
            # Note: We pass style_key as the search hint for ass_from_json
            from src.transcription import ass_from_json # Ensure import is available
            ass_from_json(
                json_path=json_file, 
                ass_path=variation_ass, 
                fontsize=base_fs * mult, 
                resolution=(1920, 1080)
                # If you modified ass_from_json to accept a song_key hint, add it here:
                # song_key=style_key 
            )

            # 2. Fast Overlay (Burn subtitles + metadata onto clean master)
            from src.render_engine import run_fast_ui_overlay # Ensure import is available
            run_fast_ui_overlay(
                clean_master_path=clean_master,
                output_path=output_mp4,
                ass_path=variation_ass,
                metadata=metadata,
                base_font_size=base_fs * mult
            )

    logger.info(f"✅ Gallery Complete! Files are in: {config['MP4_DIR']}")