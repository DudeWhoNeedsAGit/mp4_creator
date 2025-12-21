import subprocess
import numpy as np
import librosa
import json
from PIL import Image, ImageDraw
from tqdm import tqdm
from pathlib import Path

# --- THEME LOADER ---

def load_visualizer_theme(theme_name):
    """Loads a style profile from themes.json."""
    try:
        with open("themes.json", "r") as f:
            themes = json.load(f)
        return themes.get(theme_name, themes[next(iter(themes))])
    except Exception as e:
        print(f"Theme Load Warning: {e}. Falling back to default.")
        return {
            "type": "circular_bars",
            "n_bars": 72, "base_radius": 110, "bar_width": 4, 
            "sensitivity": 12.0, "decay": 0.92, "min_bar": 4,
            "palette": [[255, 230, 0], [0, 255, 255], [255, 0, 255]]
        }

# --- DRAWING STYLES (The Dispatcher Components) ---
def draw_dev_mode(draw, cfg, spectrum, smooth_amp, resolution):
    """Ultra-fast placeholder visualizer for development."""
    n_bars = 40  # Fewer bars = faster PIL processing
    spacing = resolution[0] / n_bars
    
    # We don't even use the full spectrum, just a slice for speed
    for b in range(n_bars):
        val = spectrum[b % 10] * smooth_amp * 20 
        h = max(5, min(val, 100))
        x = b * spacing
        # Plain white rectangles, no transparency, no blending
        draw.rectangle([x, resolution[1]-h, x+spacing-2, resolution[1]], fill=(255, 255, 255, 255))

def draw_neon_puls(draw, cfg, spectrum, smooth_amp, resolution):
    """Classic high-speed radial bars with a single-pass draw."""
    center_x, center_y = resolution[0] // 2, resolution[1] // 2
    n_bars = cfg.get("n_bars", 72)
    base_r = cfg.get("base_radius", 110)
    palette = cfg.get("palette", [[0, 255, 255]])
    bar_w = cfg.get("bar_width", 4)
    
    spec_len = len(spectrum) // 3
    for b in range(n_bars):
        angle = b * (2 * np.pi / n_bars) - np.pi / 2
        f_idx = int((b / n_bars) * spec_len)
        val = np.log1p(spectrum[f_idx] * cfg.get("sensitivity", 12)) * smooth_amp
        h = val * 50
        
        # Simple math, no extra layers
        x0 = center_x + base_r * np.cos(angle)
        y0 = center_y + base_r * np.sin(angle)
        x1 = center_x + (base_r + h) * np.cos(angle)
        y1 = center_y + (base_r + h) * np.sin(angle)
        
        color = tuple(palette[b % len(palette)])
        draw.line([x0, y0, x1, y1], fill=color, width=bar_w)

def draw_circular_bars(draw, cfg, spectrum, smooth_amp, resolution):
    """
    Cinematic Circular Bars: High-density needles with inner glow 
    and logarithmic frequency mapping.
    """
    center_x, center_y = resolution[0] // 2, resolution[1] // 2
    n_bars = cfg.get("n_bars", 120)  # Use more bars (120-180) for cinema look
    base_r = cfg.get("base_radius", 130)
    max_room = (min(resolution) // 2) - base_r - 10
    palette = cfg.get("palette", [[255, 255, 255]])
    bar_w = cfg.get("bar_width", 2)

    # We use a subset of the spectrum (usually the first 1/4 holds the most energy)
    spec_len = len(spectrum) // 4 

    for b in range(n_bars):
        angle = b * (2 * np.pi / n_bars) - np.pi / 2
        
        # Logarithmic indexing makes the bars react better to musical intervals
        f_idx = int(pow(b / n_bars, 1.5) * spec_len)
        f_idx = min(f_idx, spec_len - 1)
        
        mag = spectrum[f_idx]
        val = np.log1p(mag * cfg.get("sensitivity", 20)) * smooth_amp
        bar_height = max(cfg.get("min_bar", 2), min(val * 60, max_room))

        # Dynamic Color Selection
        color_idx = (b / n_bars) * (len(palette) - 1)
        c1, c2 = palette[int(color_idx)], palette[min(int(color_idx) + 1, len(palette)-1)]
        mix = color_idx - int(color_idx)
        base_color = (
            int(c1[0]*(1-mix) + c2[0]*mix),
            int(c1[1]*(1-mix) + c2[1]*mix),
            int(c1[2]*(1-mix) + c2[2]*mix)
        )

        # Math for coordinates
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x0, y0 = center_x + base_r * cos_a, center_y + base_r * sin_a
        x1, y1 = center_x + (base_r + bar_height) * cos_a, center_y + (base_r + bar_height) * sin_a

        # --- CINEMA DRAWING LAYERS ---
        # 1. Outer Glow (wider, lower opacity)
        draw.line([x0, y0, x1, y1], fill=(*base_color, 100), width=bar_w + 2)
        
        # 2. Inner Core (sharper, full opacity, slightly whiter)
        core_color = (min(255, base_color[0]+40), min(255, base_color[1]+40), min(255, base_color[2]+40), 255)
        draw.line([x0, y0, x1, y1], fill=core_color, width=bar_w)

def draw_particle_flow(draw, cfg, spectrum, smooth_amp, resolution):
    n_bars = cfg.get("n_bars", 100)
    y_base = cfg.get("y_position", 350)
    spacing = resolution[0] / n_bars
    palette = cfg.get("palette", [[255, 255, 255]])
    
    for b in range(n_bars):
        f_idx = int((b / n_bars) * (len(spectrum) // 2))
        val = np.log1p(spectrum[f_idx] * cfg.get("sensitivity", 15)) * smooth_amp
        offset = val * 100
        x = b * spacing
        color = (*palette[0], 255)
        
        # Particle dot
        r = 3 + (val * 4)
        draw.ellipse([x-r, y_base-offset-r, x+r, y_base-offset+r], fill=color)
        # Connecting line to baseline
        draw.line([x, y_base, x, y_base-offset], fill=(*palette[0], 80), width=1)

def draw_mirrored_bars(draw, cfg, spectrum, smooth_amp, resolution):
    n_bars = cfg.get("n_bars", 60)
    spacing = resolution[0] / n_bars
    palette = cfg.get("palette", [[255, 255, 255]])
    
    for b in range(n_bars):
        f_idx = int((b / n_bars) * (len(spectrum) // 4))
        val = np.log1p(spectrum[f_idx] * cfg.get("sensitivity", 10)) * smooth_amp
        h = val * 180
        x = b * spacing
        
        # Bottom bar
        draw.rectangle([x, resolution[1]-h, x+(spacing*0.7), resolution[1]], fill=(*palette[0], 255))
        # Top reflection
        if len(palette) > 1:
            draw.rectangle([x, 0, x+(spacing*0.7), h], fill=(*palette[1], 180))

def draw_floating_orbs(draw, cfg, spectrum, smooth_amp, resolution):
    """
    Cinematic Floating Orbs: Features multi-layered glow and 
    frequency-dependent vertical displacement.
    """
    n_orbs = cfg.get("n_orbs", 7)
    width, height = resolution
    spacing = width / (n_orbs + 1)
    palette = cfg.get("palette", [[255, 255, 255]])
    
    # We focus on the lower-to-mid frequencies for orbs to avoid 'jitter'
    band_size = len(spectrum) // (n_orbs + 2)

    for i in range(n_orbs):
        # 1. Frequency Analysis
        start, end = i * band_size, (i + 1) * band_size
        band_val = np.mean(spectrum[start:end])
        
        # 2. Intensity Logic
        # Sensitivity is pulled from JSON to control 'jumpiness'
        intensity = np.log1p(band_val * cfg.get("sensitivity", 18)) * smooth_amp
        
        # 3. Physics (Size and Floating Height)
        # Orbs float from a baseline of 75% screen height
        base_s = cfg.get("base_size", 40)
        size = base_s + (intensity * 150)
        float_y = (height * 0.75) - (intensity * 350) 
        x = (i + 1) * spacing
        
        # Color selection from palette
        color = palette[i % len(palette)]
        
        # 4. DRAWING LAYERS (The 'Cinema' Look)
        
        # Layer A: Large Outer Bloom (Very faint)
        bloom_size = size * 1.8
        draw.ellipse(
            [x - bloom_size, float_y - bloom_size, x + bloom_size, float_y + bloom_size], 
            fill=(*color, int(40 * intensity)) 
        )
        
        # Layer B: Inner Secondary Glow
        glow_size = size * 1.3
        draw.ellipse(
            [x - glow_size, float_y - glow_size, x + glow_size, float_y + glow_size], 
            fill=(*color, int(100 * intensity))
        )
        
        # Layer C: The Core (Solid or semi-transparent)
        # We add a tiny bit of white to the core for 'Heat'
        core_color = (
            min(255, color[0] + 50),
            min(255, color[1] + 50),
            min(255, color[2] + 50),
            255
        )
        draw.ellipse(
            [x - size, float_y - size, x + size, float_y + size], 
            fill=core_color
        )

# --- THE DISPATCHER MAP ---
VIS_MAP = {
    "circular_bars": draw_circular_bars,
    "particle_flow": draw_particle_flow,
    "mirrored_bars": draw_mirrored_bars,
    "floating_orbs": draw_floating_orbs,
    "neon_pulse": draw_neon_puls
}

# --- CORE ENGINE FUNCTIONS ---

def get_frame_count(audio_path, fps):
    return int(librosa.get_duration(path=str(audio_path)) * fps) + 1

def stream_visualizer_frames(audio_path, theme_name, fps, resolution):
    """
    Generates RGBA byte frames with a persistent Ghosting/Motion Blur engine.
    """
    cfg = load_visualizer_theme(theme_name)
    y, sr = librosa.load(audio_path, sr=None)
    hop = sr // fps
    n_frames = int(len(y) / hop) + 1
    
    # STFT for frequency data
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freq_times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=512)
    
    smooth_amp = 0.01
    draw_func = VIS_MAP.get(cfg.get("type"), draw_circular_bars)
    
    # Persistent layer for the trail effect (Motion Blur)
    trail_layer = Image.new("RGBA", resolution, (0, 0, 0, 0))

    for i in range(n_frames):
        frame_time = i / fps
        stft_idx = np.searchsorted(freq_times, frame_time)
        
        # Physics Engine (Momentum)
        start_s, end_s = i * hop, min((i+1) * hop, len(y))
        raw_amp = np.max(np.abs(y[start_s:end_s])) if start_s < len(y) else 0.01
        
        # Use decay from JSON
        decay = cfg.get("decay", 0.92)
        if raw_amp > smooth_amp:
            smooth_amp = raw_amp
        else:
            smooth_amp *= decay

        # 1. Create a fresh layer for the CURRENT frame drawing
        current_layer = Image.new("RGBA", resolution, (0, 0, 0, 0))
        draw = ImageDraw.Draw(current_layer)
        
        current_spec = S[:, stft_idx] if stft_idx < S.shape[1] else np.zeros(S.shape[0])
        
        # 2. Execute the specific drawing style (Circular, Orbs, etc.)
        draw_func(draw, cfg, current_spec, smooth_amp, resolution)

        # 3. GHOSTING ENGINE (The Cinema Secret)
        # We blend the trail with the new frame. 
        # 0.3 means 30% new frame, 70% history. Adjust to 0.2 for longer trails.
        trail_layer = Image.blend(trail_layer, current_layer, 0.3)
        
        # 4. Composite the sharp current frame over the blurred trail 
        # This keeps the 'heads' of the bars crisp while leaving a tail.
        final_frame = Image.alpha_composite(trail_layer, current_layer)
        
        yield final_frame.tobytes()

def get_audio_metadata(audio_path):
    """
    Extracts Title, Artist, and Duration.
    If tags are missing, it parses the filename (e.g., 'Artist - Title.mp3').
    """
    import subprocess
    import json
    from pathlib import Path

    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(audio_path)
    ]
    result = subprocess.check_output(cmd).decode('utf-8')
    data = json.loads(result)
    
    format_info = data.get("format", {})
    tags = format_info.get("tags", {})
    
    # 1. Try to get metadata from tags
    title = tags.get("title")
    artist = tags.get("artist")
    
    # 2. Fallback: Parse Filename if tags are missing
    # Example: "ABOMINATIO SYSTEMA _ IGNIS VERBI.mp3"
    if not title or not artist:
        filename = Path(audio_path).stem  # Removes .mp3
        if " - " in filename:
            parts = filename.split(" - ", 1)
            artist = artist or parts[0].strip()
            title = title or parts[1].strip()
        elif " _ " in filename:
            parts = filename.split(" _ ", 1)
            artist = artist or parts[0].strip()
            title = title or parts[1].strip()
        else:
            title = title or filename
            artist = artist or "Unknown Artist"

    # 3. Get Duration
    duration_secs = float(format_info.get("duration", 0))
    m, s = divmod(int(duration_secs), 60)
    duration_str = f"{m:02d}:{s:02d}"

    return {
        "title": title.upper(),
        "artist": artist.upper(),
        "duration_str": duration_str
    }

def run_integrated_render(audio_path, ass_file, bg_source, output_path, resolution, fps, theme_name):
    """
    Unified Rendering Engine.
    Feature 1: Handles Mirror-Reverse Looping for *_loopmr.mp4 files.
    """
    import json as json_lib
    import subprocess
    from pathlib import Path
    from tqdm import tqdm

    # 1. Robust Metadata Extraction
    meta = get_audio_metadata(audio_path)
    title = meta['title']
    artist = meta['artist']
    total_str = meta['duration_str']

    # 2. Logic Switches
    is_fast = theme_name in ["dev", "neon_puls"]
    preset = "ultrafast" if is_fast else "veryfast"
    crf = "23" if is_fast else "18"
    vis_filter = "null" if is_fast else "rgbashift=rh=1:bv=-1"
    
    # Feature 1: Check for Mirror-Reverse Loop requirement
    is_video = bg_source.suffix.lower() == ".mp4"
    is_mirror_loop = "_loopmr" in bg_source.name.lower()

    # 3. Setup Paths and Escaping
    res_str = f"{resolution[0]}x{resolution[1]}"
    ass_path_fixed = str(ass_file.absolute()).replace("\\", "/").replace(":", "\\:")
    total_frames = get_frame_count(audio_path, fps)

    # Escape strings for FFmpeg Drawtext (Moved outside f-string to avoid SyntaxError)
    safe_title = title.replace(":", "\\:")
    safe_author = f"by {artist}".replace(":", "\\:")
    safe_total_escaped = total_str.replace(":", "\\:")
    
    line_title = f"{safe_title}"
    line_author = f"{safe_author}"
    line_timer  = f"%{{pts\\:gmtime\\:0\\:%M\\\\\\:%S}} / {safe_total_escaped}"

    # 4. Build FFmpeg Command Base
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]

    # Input 0: Background
    if is_video:
        # We handle looping inside filter_complex for mirror-looping, 
        # but for standard video we use stream_loop
        if not is_mirror_loop:
            cmd += ["-stream_loop", "-1"]
        cmd += ["-i", str(bg_source)]
    else:
        cmd += ["-framerate", str(fps), "-loop", "1", "-i", str(bg_source)]

    # Input 1: Visualizer Pipe, Input 2: Audio
    cmd += [
        "-f", "rawvideo", "-vcodec", "rawvideo", "-s", res_str,
        "-pix_fmt", "rgba", "-framerate", str(fps), "-i", "-",
        "-i", str(audio_path)
    ]

    # 5. Build the Filter Complex
    # Define Background Logic (Feature 1: Looppool)
    if is_video and is_mirror_loop:
        bg_chain = (
            f"[0:v]fps={fps},scale={resolution[0]}:{resolution[1]},setsar=1[base];"
            f"[base]split[fwd][prep_bwd];"
            f"[prep_bwd]reverse[bwd];"
            f"[fwd][bwd]concat=n=2:v=1:a=0,loop=-1:size=600[bg];"
        )
    else:
        bg_chain = f"[0:v]fps={fps},scale={resolution[0]}:{resolution[1]},setsar=1[bg];"

    # Main linear pipe: BG -> Overlay Vis -> Draw HUD -> Add Subtitles
    cmd += [
        "-filter_complex",
        f"{bg_chain}"
        f"[1:v]{vis_filter}[vis];"
        f"[bg][vis]overlay=0:0:shortest=1[v1];"
        f"[v1]drawtext=text='{line_title}':fontcolor=white:fontsize=14:x=20:y=(h/2)-40:shadowcolor=black:shadowx=2:shadowy=2,"
        f"drawtext=text='{line_author}':fontcolor=white:fontsize=12:x=20:y=(h/2)-20:alpha=0.6:shadowcolor=black:shadowx=2:shadowy=2,"
        f"drawtext=text='{line_timer}':fontcolor=white:fontsize=10:x=20:y=(h/2):alpha=0.8:shadowcolor=black:shadowx=1:shadowy=1,"
        f"ass='{ass_path_fixed}'[outv]",
        
        "-map", "[outv]", 
        "-map", "2:a",
        "-c:v", "libx264", "-preset", preset, "-crf", crf, "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-shortest", str(output_path)
    ]

    # 6. Execution Process
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    try:
        with tqdm(total=total_frames, desc=f"🎬 Rendering {theme_name}", unit="frame") as pbar:
            for frame_bytes in stream_visualizer_frames(audio_path, theme_name, fps, resolution):
                process.stdin.write(frame_bytes)
                pbar.update(1)
    except BrokenPipeError:
        print("\n[!] FFmpeg pipe closed unexpectedly. Check filter strings.")
    finally:
        process.stdin.close()
        process.wait()

    print(f"\n[SUCCESS] Video rendered at: {output_path}")