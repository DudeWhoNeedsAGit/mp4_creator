Here is a structured backlog item you can copy into your project management tool (or a TODO.md file) to track the implementation of these advanced subtitle effects.

🛠️ Backlog Item: Enhance Subtitle Visual Engine (ASS Override Tags)
Priority: Low/Medium (Visual Polish)

Status: Discovery

📝 Description
Currently, our subtitle engine (ass_from_json) uses static styles defined in the header. To increase production value and match specific musical themes (Goth, Industrial, Synthwave), we need to transition from static styles to inline override tags. This allows for word-level animations, glows, and 3D transformations.

🎯 Objective
Upgrade the lines.append logic in src/transcription.py to support dynamic visual feedback during the karaoke phase.

🧪 Proposed Effects to Test
The "Active Pop": Use \fscx and \fscy to make the current word 10% larger while active.

Neon Halo: Implement \be (Blur Edges) combined with a thicker \bord (Border) to create a glowing neon effect.

Industrial Distortion: Apply \fax (Shear) to specific themes to give the text a "glitched" or slanted digital look.

3D Perspective: Experiment with \fry (Y-axis rotation) for "wall-style" lyrics.

🛠️ Technical Implementation Notes
Target File: src/transcription.py -> ass_from_json()

Logic Change: Modify the payload.append loop.

Code Snippet Example:

Python

# Dynamic "Pop" effect template
payload.append(f"{{\\fscx110\\fscy110\\t(0,{dur_half},\\fscx100\\fscy100)\\kf{dur_cs}}}{word} ")
✅ Definition of Done
[ ] Subtitles remain synchronized with the audio.

[ ] Effects are toggleable via mp3_configs.json (e.g., "effect": "neon_glow").

[ ] No significant impact on FFmpeg render times.

[ ] Text remains legible across all tested resolutions (480p/1080p).