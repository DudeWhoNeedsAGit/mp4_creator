Here is the detailed backlog item for your Vertical/Shorts integration. I have structured this to match your current processor.py logic, specifically focusing on the Two-Pass Master/Overlay system.

🚀 Backlog Item: 002 - Vertical (9:16) Shorts Support
Overview
Implement an automated workflow for generating 1080x1920 vertical videos. This leverages the existing clean_master architecture to allow fast iteration on subtitle styles while ensuring the visualizer is positioned correctly for mobile UI "Safe Zones."

High-Level Integration Points
1. Argument Handling (main.py)
Task: Add --vertical flag.

Logic: If --vertical is toggled:

Set CONFIG['VIDEO_RES'] to (1080, 1920).

Update res_tag to include _shorts (e.g., _shorts_1080p) to prevent filename collisions with horizontal renders.

2. Adaptive Background Discovery (processor.py)
Task: Update bg_source selection logic.

Priority Search:

{basename}_vertical_loopmr.mp4 (Mirror Loop Vertical)

{basename}_vertical.mp4 (Standard Vertical)

{basename}_loopmr.mp4 (Horizontal - will require FFmpeg pad/crop filter)

3. Geometry & Safe Zones (render_engine.py)
Task: Coordinate Injection for Numba.

Vertical Center: Unlike horizontal (middle), the vertical center should be offset to (540, 850) to stay clear of the bottom-heavy YouTube Shorts UI (Channel name, Subscribe button, Song info).

Safe Zone Reference:

4. Scale-Aware Subtitles (transcription.py)
Task: Vertical alignment shift.

Logic: When do_upscale and is_vertical are both true, the .ass generation must shift the PlayResY and vertical margins to ensure text appears in the "sweet spot" (middle-bottom but above the UI).

Implementation Plan (Code Snippets)
Phase A: main.py Update
Python

# main.py
parser.add_argument('--vertical', action='store_true', help="Render in 9:16 for Shorts/TikTok")

# Inside main() logic:
if args.vertical:
    CONFIG['VIDEO_RES'] = (1080, 1920)
    res_tag = "_shorts_1080p" if args.upscale else "_shorts_480p"
Phase B: processor.py Geometry
Python

# processor.py inside process_file()
is_vertical = getattr(args, 'vertical', False)

if is_vertical:
    render_res = (1080, 1920)
    render_center = (540, 850) # Shifted up from 960 to avoid UI
else:
    render_res = config['VIDEO_RES']
    render_center = (render_res[0] // 2, render_res[1] // 2)
Phase C: render_engine.py GPU Update
Python

# Update function signature
def run_integrated_render_gpu(..., center_override=None):
    # Use center_override if provided, else calculate default center
    c_x, c_y = center_override if center_override else (width // 2, height // 2)
Definition of Done
[ ] Script successfully detects _vertical_loopmr.mp4 and uses 1080x1920 resolution.

[ ] Visualizer circle renders at the injected center_override coordinates.

[ ] Subtitles are visible and not obscured by the "Subscribe" button on a real mobile device.

[ ] Reusing a "Clean Vertical Master" works with the run_fast_ui_overlay path.