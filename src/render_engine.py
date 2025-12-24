import subprocess
import numpy as np
import librosa
import json
from PIL import Image, ImageDraw
from tqdm import tqdm
from pathlib import Path
from numba import njit, prange, uint8
import logging
logger = logging.getLogger(__name__)

def extract_palette_from_bg(bg_path, n_colors=5, min_distance=180, min_hue_diff=60):
    """Prioritizes dominant, enforces distance + hue separation."""
    from PIL import Image
    import numpy as np
    import cv2
    from sklearn.cluster import KMeans
    import colorsys
    import logging

    logger = logging.getLogger(__name__)
    logger.info(f"🎨 Quick palette extraction: {bg_path.name}")

    if bg_path.suffix.lower() in [".mp4", ".mov"]:
        cap = cv2.VideoCapture(str(bg_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            frame = np.zeros((480, 854, 3), dtype=np.uint8)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.cvtColor(cv2.imread(str(bg_path)), cv2.COLOR_BGR2RGB)

    small = cv2.resize(image, (50, 50))
    pixels = small.reshape(-1, 3)

    kmeans = KMeans(n_clusters=n_colors*2, n_init=5, random_state=42)
    kmeans.fit(pixels)
    centers = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    counts = np.bincount(labels)
    order = np.argsort(-counts)

    boosted = []
    hues = []
    for idx in order:
        c = centers[idx]
        r,g,b = c/255.0
        h,s,v = colorsys.rgb_to_hsv(r,g,b)
        s = max(s, 0.85)
        v = max(v, 0.9)
        r,g,b = colorsys.hsv_to_rgb(h,s,v)
        boosted.append(np.array([int(r*255), int(g*255), int(b*255)], dtype=np.uint8))
        hues.append(h * 360)

    final = [boosted[0]]  # Force dominant
    final_hues = [hues[0]]
    logger.debug(f"Force accepted dominant: {boosted[0].tolist()} (hue {hues[0]:.0f}°)")

    for i in range(1, len(boosted)):
        c = boosted[i]
        h = hues[i]
        logger.debug(f"Checking candidate: {c.tolist()} (hue {h:.0f}°)")
        if len(final) >= n_colors: break
        dist_ok = all(np.linalg.norm(c - existing) >= min_distance for existing in final)
        hue_ok = all(min(abs(h - fh) % 360, 360 - abs(h - fh)) >= min_hue_diff for fh in final_hues)
        if dist_ok and hue_ok:
            logger.debug(f"Accepted: {c.tolist()}")
            final.append(c)
            final_hues.append(h)
        else:
            logger.debug(f"Rejected (dist/hue): distances {[np.linalg.norm(c - e) for e in final]}, hue diffs {[min(abs(h - fh) % 360, 360 - abs(h - fh)) for fh in final_hues]}")

    fallback = np.array([[0,255,255], [255,0,255], [0,255,127], [255,105,180], [0,128,255]], dtype=np.uint8)
    fallback_hues = [180, 300, 150, 330, 210]  # Approx hues
    i = 0
    while len(final) < n_colors and i < len(fallback)*2:
        f = fallback[i % len(fallback)]
        fh = fallback_hues[i % len(fallback)]
        logger.debug(f"Checking fallback: {f.tolist()} (hue {fh}°)")
        dist_ok = all(np.linalg.norm(f - existing) >= min_distance for existing in final)
        hue_ok = all(min(abs(fh - existing_h) % 360, 360 - abs(fh - existing_h)) >= min_hue_diff for existing_h in final_hues)
        if dist_ok and hue_ok:
            logger.debug("Accepted fallback")
            final.append(f)
            final_hues.append(fh)
        i += 1

    final_palette = np.array(final[:n_colors], dtype=np.uint8)

    def approx_name(rgb):
        r,g,b = rgb
        if r > 200 and g < 100 and b < 100: return "red"
        if r == g == 255: return "cyan"
        if r == b == 255: return "magenta"
        if g == b == 255: return "teal"
        if b == 255: return "blue"
        return "custom"

    names = [f"{approx_name(c)} ({c.tolist()})" for c in final_palette]
    logger.info(f"✅ Palette: {', '.join(names)}")

    return final_palette

# --- THEME LOADER ---

def load_visualizer_theme(theme_name):
    try:
        with open("themes.json", "r") as f:
            themes = json.load(f)
        
        # Check if requested theme exists
        if theme_name in themes:
            return themes[theme_name]
        
        # Log that it's missing and fall back to a known safe theme
        print(f"Theme '{theme_name}' not found. Falling back to default.")
        return themes.get("default", themes[next(iter(themes))])
        
    except Exception as e:
        print(f"Theme Load Warning: {e}. Falling back to hardcoded default.")
        return {
            "type": "neon_pulse_numba",
            "n_bars": 80, "base_radius": 110, "bar_width": 4, 
            "sensitivity": 1.5, "decay": 0.8, "min_bar": 4,
            "palette": [[255, 230, 0], [0, 255, 255], [255, 0, 255]]
        }

# --- NEW: KINETIC NEBULA KERNELS ---
@njit(fastmath=True)
def lerp_color(palette, t):
    """Calculates the smooth color transition at position t (0.0 to 1.0)."""
    num_colors = len(palette)
    # Map t to the palette index space
    idx = t * (num_colors - 1)
    i = int(idx)
    f = idx - i  # Fraction between color i and i+1
    
    c1 = palette[i]
    c2 = palette[min(i + 1, num_colors - 1)]
    
    r = uint8(c1[0] * (1 - f) + c2[0] * f)
    g = uint8(c1[1] * (1 - f) + c2[1] * f)
    b = uint8(c1[2] * (1 - f) + c2[2] * f)
    return r, g, b


@njit(fastmath=True)
def draw_line_fast(pixels, x0, y0, x1, y1, width, r, g, b):
    h_img, w_img, _ = pixels.shape
    dx, dy = x1 - x0, y1 - y0
    length = np.sqrt(dx*dx + dy*dy)
    if length == 0: return
    
    ux, uy = dx/length, dy/length
    px, py = -uy * (width/2), ux * (width/2)
    
    min_x, max_x = max(0, int(min(x0, x1) - abs(px))), min(w_img-1, int(max(x0, x1) + abs(px)))
    min_y, max_y = max(0, int(min(y0, y1) - abs(py))), min(h_img-1, int(max(y0, y1) + abs(py)))
    
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            pax, pay = x - x0, y - y0
            t = (pax * ux + pay * uy) / length
            if 0 <= t <= 1:
                dist = abs(pax * (-uy) + pay * ux)
                if dist <= width / 2:
                    pixels[y, x, 0], pixels[y, x, 1], pixels[y, x, 2], pixels[y, x, 3] = r, g, b, 255

@njit(parallel=True, fastmath=True)
def render_neon_circle_crisp(pixels, n_bars, base_r, bar_heights, colors, center, bar_width):
    cx, cy = center
    # Parallelize over bars, not pixels (Much faster for sparse drawings)
    for b in prange(n_bars):
        angle = b * (2 * np.pi / n_bars) - np.pi / 2
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        # Start and End points
        x0 = cx + base_r * cos_a
        y0 = cy + base_r * sin_a
        x1 = cx + (base_r + bar_heights[b]) * cos_a
        y1 = cy + (base_r + bar_heights[b]) * sin_a
        
        # Draw the main neon bar
        draw_line_fast(pixels, x0, y0, x1, y1, bar_width, colors[b,0], colors[b,1], colors[b,2])

@njit(parallel=True, fastmath=True)
def render_circular_bars_numba(pixels, n_bars, base_r, bar_heights, palette, center, bar_w):
    cx, cy = center
    for b in prange(n_bars):
        t = b / n_bars
        r, g, b_v = lerp_color(palette, t)
        
        angle = b * (2 * np.pi / n_bars) - np.pi / 2
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        h = bar_heights[b]
        
        x0, y0 = cx + base_r * cos_a, cy + base_r * sin_a
        x1, y1 = cx + (base_r + h) * cos_a, cy + (base_r + h) * sin_a

        # --- CINEMA DRAWING LAYERS ---
        # 1. Outer Glow (Wider, dimmer)
        draw_line_fast(pixels, x0, y0, x1, y1, bar_w + 2, r * 0.4, g * 0.4, b_v * 0.4)
        
        # 2. Inner Core (Sharper, whiter)
        r_c = min(255.0, float(r) + 40.0)
        g_c = min(255.0, float(g) + 40.0)
        b_c = min(255.0, float(b_v) + 40.0)
        draw_line_fast(pixels, x0, y0, x1, y1, bar_w, r_c, g_c, b_c)

@njit(parallel=True, fastmath=True)
def render_neon_puls_numba(pixels, n_bars, base_r, bar_heights, palette, center, bar_w):
    """Numba port with smooth cyclic interpolation (last connects to first)."""
    cx, cy = center
    n_colors = len(palette)
    
    for b in prange(n_bars):
        # Cyclic interpolation: t from 0 to 1, wraps seamlessly
        t = b / n_bars
        idx = t * n_colors
        color_idx1 = int(idx) % n_colors
        color_idx2 = (color_idx1 + 1) % n_colors
        frac = idx - int(idx)
        
        r = int(palette[color_idx1][0] * (1 - frac) + palette[color_idx2][0] * frac)
        g = int(palette[color_idx1][1] * (1 - frac) + palette[color_idx2][1] * frac)
        b_v = int(palette[color_idx1][2] * (1 - frac) + palette[color_idx2][2] * frac)
        
        angle = b * (2 * np.pi / n_bars) - np.pi / 2
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        h = bar_heights[b]
        
        x0, y0 = cx + base_r * cos_a, cy + base_r * sin_a
        x1, y1 = cx + (base_r + h) * cos_a, cy + (base_r + h) * sin_a
        
        draw_line_fast(pixels, x0, y0, x1, y1, bar_w, r, g, b_v)

@njit(fastmath=True)
def draw_line_kernel(img, x0, y0, x1, y1, color, width):
    """Pure machine-code DDA Line drawing."""
    dx = x1 - x0
    dy = y1 - y0
    steps = max(abs(dx), abs(dy))
    if steps < 1: return
    
    x_inc = dx / steps
    y_inc = dy / steps
    x, y = x0, y0
    
    h, w, _ = img.shape
    r, g, b, a = color

    for _ in range(int(steps)):
        ix, iy = int(x), int(y)
        if 0 <= ix < w and 0 <= iy < h:
            # Draw width-wise to simulate thickness
            for ww in range(width):
                if 0 <= iy + ww < h:
                    img[iy + ww, ix, 0] = r
                    img[iy + ww, ix, 1] = g
                    img[iy + ww, ix, 2] = b
                    img[iy + ww, ix, 3] = a
        x += x_inc
        y += y_inc

@njit(parallel=True)
def apply_ghosting_and_composite(trail, current, decay=0.7):
    h, w, _ = trail.shape
    for y in prange(h):
        for x in range(w):
            if current[y, x, 3] > 0:
                trail[y, x, :] = current[y, x, :]
            else:
                trail[y, x, 0] = uint8(trail[y, x, 0] * decay)
                trail[y, x, 1] = uint8(trail[y, x, 1] * decay)
                trail[y, x, 2] = uint8(trail[y, x, 2] * decay)
                # Purge shadows
                if trail[y, x, 0] < 12 and trail[y, x, 1] < 12 and trail[y, x, 2] < 12:
                    trail[y, x, :] = 0

@njit(fastmath=True)
def draw_line_numba(img, x0, y0, x1, y1, r, g, b, a, width):
    """High-speed DDA line drawing kernel."""
    dx = x1 - x0
    dy = y1 - y0
    steps = max(abs(dx), abs(dy))
    if steps < 1: return
    
    x_inc = dx / steps
    y_inc = dy / steps
    x, y = x0, y0
    
    h, w, _ = img.shape
    
    for _ in range(int(steps)):
        ix, iy = int(x), int(y)
        if 0 <= ix < w and 0 <= iy < h:
            # Simple width expansion
            for ww in range(width):
                if 0 <= iy + ww < h:
                    img[iy + ww, ix, 0] = r
                    img[iy + ww, ix, 1] = g
                    img[iy + ww, ix, 2] = b
                    img[iy + ww, ix, 3] = a
        x += x_inc
        y += y_inc

@njit(parallel=True, fastmath=True)
def render_cyber_horizon(pixels, n_bars, bar_heights, colors, center):
    h_screen, w_screen, _ = pixels.shape
    cy = h_screen // 2  
    
    total_bar_w = w_screen / n_bars
    gap = 1
    bar_w = max(1, int(total_bar_w) - gap)
    
    # NEW: Global Opacity Control (0.0 to 1.0)
    # You can also pass this as an argument from cfg
    opacity = 0.7 

    for b in prange(n_bars):
        x_start = int(b * total_bar_w)
        x_end = x_start + bar_w
        if x_end >= w_screen: continue
        
        # Scaling the height down for a cleaner look
        height = bar_heights[b] * 1.2 
        if height < 2: continue

        r, g, b_val = colors[b, 0], colors[b, 1], colors[b, 2]

        for x in range(x_start, x_end):
            # 1. TOP BAR (The "Sky")
            y_top_start = cy - int(height)
            for y in range(max(0, y_top_start), cy):
                # Subtle curve: peaks are softer, base is more solid
                rel_pos = (cy - y) / height
                intensity = (1.0 - rel_pos**2) * opacity 
                
                # Additive-lite blending
                pixels[y, x, 0] = uint8(min(255, pixels[y, x, 0] + r * intensity))
                pixels[y, x, 1] = uint8(min(255, pixels[y, x, 1] + g * intensity))
                pixels[y, x, 2] = uint8(min(255, pixels[y, x, 2] + b_val * intensity))
                pixels[y, x, 3] = 255

            # 2. BOTTOM BAR (The "Reflection")
            y_bot_end = cy + int(height * 0.6) # Shorter reflection
            for y in range(cy, min(h_screen, y_bot_end)):
                dist_from_horizon = (y - cy)
                # Faster fade out for reflection
                fade = (1.0 - (dist_from_horizon / (height * 0.6)))**2 * 0.4
                
                if y % 2 == 0: fade *= 0.3 # Stronger scanline effect

                pixels[y, x, 0] = uint8(min(255, pixels[y, x, 0] + r * fade))
                pixels[y, x, 1] = uint8(min(255, pixels[y, x, 1] + g * fade))
                pixels[y, x, 2] = uint8(min(255, pixels[y, x, 2] + b_val * fade))
                pixels[y, x, 3] = 255

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
# Note: We keep the old ones for compatibility, 
# but Numba will now handle the "heavy" types.
VIS_MAP = {
    "circular_bars": render_circular_bars_numba, # Your existing Numba bar logic
    "cyber_horizon": render_cyber_horizon, 
    "neon_circle": render_neon_circle_crisp, # The new Field Kernel
    "neon_pulse_numba": render_circular_bars_numba,
    "neon_pulse_old": draw_neon_puls,
    "particle_flow": draw_particle_flow, # Note: These PIL ones will be slower
    "floating_orbs": draw_floating_orbs
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

# --- STEP 2: THE MAIN STREAMER ---
def stream_visualizer_frames_numba(audio_path, theme_name, fps, resolution, limit_frames=None, override_palette=None):
    import librosa
    import numpy as np
    
    cfg = load_visualizer_theme(theme_name)
    width, height = resolution
    center = (width // 2, height // 2)
    
    # 1. Load Audio
    y, sr = librosa.load(audio_path, sr=None)
    hop = sr // fps
    n_frames = int(len(y) / hop) + 1
    if limit_frames:
        n_frames = min(n_frames, limit_frames)
    
    # 2. Spectral Analysis (Matching your 2048 n_fft)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freq_times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=512)
    
    # 3. Buffers
    trail_buffer = np.zeros((height, width, 4), dtype=np.uint8)
    current_frame_buffer = np.zeros((height, width, 4), dtype=np.uint8)
    
    # 4. Config & Dynamic Palette Setup
    n_bars = cfg.get("n_bars", 72)
    base_r = cfg.get("base_radius", 110)
    bar_w = cfg.get("bar_width", 4)
    
    # Use the dynamic palette if it exists, else fall back to theme
    if override_palette is not None:
        palette = override_palette
    else:
        palette = np.array(cfg.get("palette", [[0, 255, 255]]), dtype=np.uint8)
        
    sensitivity = cfg.get("sensitivity", 12.0)
    decay_rate = cfg.get("decay", 0.85)
    smooth_amp = 0.01

    # 5. Main Rendering Loop
    for i in range(n_frames):
        stft_idx = np.searchsorted(freq_times, i / fps)
        current_spec = S[:, stft_idx] if stft_idx < S.shape[1] else np.zeros(S.shape[0])
        
        # Physics (Momentum)
        start_s, end_s = i * hop, min((i+1) * hop, len(y))
        raw_amp = np.max(np.abs(y[start_s:end_s])) if start_s < len(y) else 0.01
        
        if raw_amp > smooth_amp: 
            smooth_amp = raw_amp
        else: 
            smooth_amp *= decay_rate

        # The "Good" Math: Linear Mapping & 50x Height
        bar_heights = np.zeros(n_bars, dtype=np.float32)
        spec_len = len(current_spec) // 3
        for b in range(n_bars):
            f_idx = int((b / n_bars) * spec_len)
            f_idx = min(f_idx, spec_len - 1)
            
            # Using log1p scaling for that punchy reaction
            val = np.log1p(current_spec[f_idx] * sensitivity) * smooth_amp
            bar_heights[b] = val * 50

        # Draw Frame
        current_frame_buffer.fill(0)
        
        # Delegation: Ensure this matches your VIS_MAP or direct call
        # Using draw_neon_puls as per your previous preference for smoothness
        render_neon_puls_numba(current_frame_buffer, n_bars, base_r, bar_heights, palette, center, bar_w)

        # The "Anti-Shadow" Compositor
        # This keeps the motion blur but kills the dark gray residue
        apply_ghosting_and_composite(trail_buffer, current_frame_buffer, decay=0.7)
        
        yield trail_buffer.tobytes()

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

def run_integrated_render_gpu(audio_path, ass_file, bg_source, output_path, resolution, fps, theme_name, limit_frames=None):    
    import subprocess
    from pathlib import Path
    from tqdm import tqdm
    import numpy as np

    # --- NEW: DYNAMIC PALETTE EXTRACTION ---
    # We extract the palette once here
    dynamic_palette = extract_palette_from_bg(bg_source, n_colors=3)
    # ---------------------------------------

    # 1. Metadata & Settings
    meta = get_audio_metadata(audio_path)
    res_str = f"{resolution[0]}x{resolution[1]}"
    total_frames = get_frame_count(audio_path, fps)
    if limit_frames:
        total_frames = min(total_frames, limit_frames)

    if ass_file is not None:
        ass_path_fixed = str(Path(ass_file).absolute()).replace("\\", "/").replace(":", "\\:")
        ass_filter = f",ass='{ass_path_fixed}'"
    else:
        ass_filter = ""

    # 2. ESCAPE STRINGS
    safe_title = meta['title'].replace(":", "\\:")
    safe_author = f"by {meta['artist']}".replace(":", "\\:")
    safe_total = meta['duration_str'].replace(":", "\\:")
    line_timer = f"%{{pts\\:gmtime\\:0\\:%M\\\\\\:%S}} / {safe_total}"

    # 3. BUILD COMMAND
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]

    is_video = bg_source.suffix.lower() == ".mp4"
    is_mirror_loop = "_loopmr" in bg_source.name.lower()

    if is_video:
        if not is_mirror_loop:
            cmd += ["-stream_loop", "-1"]
        cmd += ["-an"]

    cmd += ["-i", str(bg_source)]

    cmd += [
        "-f", "rawvideo", "-vcodec", "rawvideo", "-s", res_str,
        "-pix_fmt", "rgba", "-framerate", str(fps), "-i", "-",
        "-i", str(audio_path)
    ]

    # 4. OPTIMIZED FILTER COMPLEX
    if is_video and is_mirror_loop:
        bg_chain = (
            f"[0:v]fps={fps},scale={resolution[0]}:{resolution[1]}:force_original_aspect_ratio=increase,crop={resolution[0]}:{resolution[1]},setsar=1[base];"
            f"[base]split[fwd][prep_bwd];"
            f"[prep_bwd]reverse[bwd];"
            f"[fwd][bwd]concat=n=2:v=1:a=0,loop=-1:size=600[bg];"
        )
    else:
        bg_chain = f"[0:v]fps={fps},scale={resolution[0]}:{resolution[1]}:force_original_aspect_ratio=increase,crop={resolution[0]}:{resolution[1]},setsar=1[bg];"
    
    if ass_file is not None:
        text_overlays = (
            f"[v1]drawtext=text='{safe_title}':fontcolor=white:fontsize=14:x=20:y=(h/2)-40:shadowcolor=black@0.5:shadowx=2:shadowy=2,"
            f"drawtext=text='{safe_author}':fontcolor=white:fontsize=12:x=20:y=(h/2)-20:alpha=0.6:shadowcolor=black@0.5:shadowx=2:shadowy=2,"
            f"drawtext=text='{line_timer}':fontcolor=white:fontsize=10:x=20:y=(h/2):alpha=0.8:shadowcolor=black@0.4:shadowx=1:shadowy=1"
            f"{ass_filter}[outv]"
        )
    else:
        text_overlays = "[v1]copy[outv]"

    cmd += [
        "-filter_complex",
        f"{bg_chain}[1:v]format=rgba[vis];[bg][vis]overlay=0:0:shortest=1[v1];{text_overlays}",
        "-map", "[outv]",
        "-map", "2:a",
        "-c:v", "h264_nvenc", 
        "-preset", "p4", "-tune", "hq", "-rc", "vbr", "-cq", "20", "-pix_fmt", "yuv420p", 
        "-c:a", "aac", "-b:a", "192k", "-shortest", str(output_path)
    ]

    # 5. EXECUTION
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    try:
        with tqdm(total=total_frames, desc=f"🎬 Rendering {theme_name}", unit="frame") as pbar:
            # PASS THE DYNAMIC PALETTE into the streamer
            # Note: You'll need to update your streamer to accept 'override_palette'
            frames_gen = stream_visualizer_frames_numba(
                audio_path, theme_name, fps, resolution, 
                override_palette=dynamic_palette
            )
            
            for i, frame_bytes in enumerate(frames_gen):
                if limit_frames and i >= limit_frames:
                    break
                process.stdin.write(frame_bytes)
                pbar.update(1)
    finally:
        process.stdin.close()
        process.wait()

def run_fast_ui_overlay(clean_master_path, output_path, ass_path, metadata, base_font_size):
    import subprocess
    from pathlib import Path
    import logging

    logger = logging.getLogger("subtitle_gen")

    # Scale metadata fonts for 1080p
    title_fs = int(base_font_size * 1.0)
    artist_fs = int(base_font_size * 0.9)
    timer_fs = int(base_font_size * 0.8)
    
    # 1. Escape Metadata Text for drawtext
    # drawtext requires ':' and "'" to be escaped
    safe_title = metadata.get('title', 'Unknown').replace(":", "\\:").replace("'", "\\'")
    safe_artist = f"by {metadata.get('artist', 'Unknown')}".replace(":", "\\:").replace("'", "\\'")
    safe_total = metadata.get('duration_str', '00\\:00').replace(":", "\\:")
    line_timer = f"%{{pts\\:gmtime\\:0\\:%M\\\\\\:%S}} / {safe_total}"
    
    # 2. THE CRITICAL PATH FIX:
    # FFmpeg 'ass' filter needs the path to be escaped for the filter-graph
    # We use .absolute() and then handle the string replacements
    ass_path_str = str(Path(ass_path).absolute()).replace("\\", "/").replace(":", "\\:")
    # Wrap in single quotes to handle spaces or dots in filenames
    ass_filter = f"ass='{ass_path_str}'"

    filters = [
        f"drawtext=text='{safe_title}':fontcolor=white:fontsize={title_fs}:x=40:y=(h/2)-100:shadowcolor=black@0.6:shadowx=4:shadowy=4",
        f"drawtext=text='{safe_artist}':fontcolor=white:fontsize={artist_fs}:x=40:y=(h/2)-60:alpha=0.8:shadowcolor=black@0.6:shadowx=3:shadowy=3",
        f"drawtext=text='{line_timer}':fontcolor=white:fontsize={timer_fs}:x=40:y=(h/2)-20:alpha=0.8:shadowcolor=black@0.4:shadowx=2:shadowy=2",
        ass_filter
    ]

    cmd = [
        "ffmpeg", "-y", 
        "-i", str(clean_master_path),
        "-vf", ",".join(filters),
        "-c:v", "h264_nvenc", "-preset", "p2", "-cq", "23", 
        "-c:a", "copy",
        str(output_path)
    ]
    
    try:
        # Changed to capture stderr to help diagnose if it fails again
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg Overlay Failed!")
        logger.error(f"Command: {' '.join(cmd)}")
        logger.error(f"Error Output: {e.stderr}")
        raise e