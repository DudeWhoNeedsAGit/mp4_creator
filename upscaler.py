import torch
import cv2
import os
import subprocess
import threading
import time
from queue import Queue
from pathlib import Path
from tqdm import tqdm
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

def get_optimal_settings():
    """Detects hardware to set tile size and precision."""
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram < 6:
            return 400, True # Low VRAM
        return 0, True # Auto tile, FP16
    return 512, False

def frame_reader(cap, queue, stop_event):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            queue.put(None)
            break
        queue.put(frame)
        while queue.qsize() > 50 and not stop_event.is_set():
            threading.Event().wait(0.005)

def frame_writer(output_queue, ffmpeg_process, stop_event):
    while not stop_event.is_set():
        if output_queue.empty():
            threading.Event().wait(0.005)
            continue
        frame = output_queue.get()
        if frame is None: break
            
        if frame.shape[0] != 1080 or frame.shape[1] != 1920:
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
            
        try:
            ffmpeg_process.stdin.write(frame.tobytes())
        except (BrokenPipeError, IOError):
            break

def run_upscale(input_path, output_path, ass_path=None, metadata=None, limit_frames=None, base_font_size=20):
    start_time = time.time()
    input_path, output_path = str(input_path), str(output_path)
    tile_size, use_fp16 = get_optimal_settings()
    
    # 1. MODEL INIT
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    upscaler = RealESRGANer(
        scale=2, model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        model=model, tile=tile_size, half=use_fp16, device='cuda'
    )

    # 2. VIDEO SETUP
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if limit_frames: total_frames = min(total_frames, limit_frames)
    
    input_queue = Queue(maxsize=50); output_queue = Queue(maxsize=30); stop_event = threading.Event()
    temp_silent_video = output_path.replace(".mp4", "_silent_tmp.mp4")

    # 3. 1080p SCALED OVERLAYS (Title, Artist, Timer)
    title_fs = int(base_font_size * 1.8)   # Proportional scaling
    artist_fs = int(base_font_size * 1.4)
    timer_fs = int(base_font_size * 1.4)
    pos_x = 40 
    
    video_filters = ["format=bgr24", "scale=1920:1080:flags=lanczos"]
    
    if metadata:
        safe_title = metadata.get('title', 'Unknown').replace(":", "\\:")
        safe_artist = f"by {metadata.get('artist', 'Unknown')}".replace(":", "\\:")
        safe_total = metadata.get('duration_str', '00\\:00').replace(":", "\\:")
        # Re-implementing the pts timer for the 1080p pass
        line_timer = f"%{{pts\\:gmtime\\:0\\:%M\\\\\\:%S}} / {safe_total}"
        
        video_filters.append(
            f"drawtext=text='{safe_title}':fontcolor=white:fontsize={title_fs}:x={pos_x}:y=(h/2)-100:"
            f"shadowcolor=black@0.6:shadowx=4:shadowy=4:fix_bounds=1"
        )
        video_filters.append(
            f"drawtext=text='{safe_artist}':fontcolor=white:fontsize={artist_fs}:x={pos_x}:y=(h/2)-60:"
            f"alpha=0.8:shadowcolor=black@0.6:shadowx=3:shadowy=3:fix_bounds=1"
        )
        video_filters.append(
            f"drawtext=text='{line_timer}':fontcolor=white:fontsize={timer_fs}:x={pos_x}:y=(h/2)-20:"
            f"alpha=0.8:shadowcolor=black@0.4:shadowx=2:shadowy=2"
        )

    if ass_path and os.path.exists(ass_path):
        ass_fixed = str(Path(ass_path).absolute()).replace("\\", "/").replace(":", "\\:")
        video_filters.append(f"ass='{ass_fixed}'")
    
    filter_str = ",".join(video_filters)

    # 4. FFMPEG & THREADS (rest of logic same as previous full code)
    command = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', '1920x1080',
        '-pix_fmt', 'bgr24', '-r', str(fps), '-i', '-',
        '-vf', filter_str,
        '-c:v', 'h264_nvenc', '-preset', 'p4', '-cq', '20', '-pix_fmt', 'yuv420p', 
        temp_silent_video
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # 5. START THREADS
    reader_t = threading.Thread(target=frame_reader, args=(cap, input_queue, stop_event))
    writer_t = threading.Thread(target=frame_writer, args=(output_queue, process, stop_event))
    reader_t.start()
    writer_t.start()

    # 6. MAIN GPU INFERENCE LOOP
    processed_count = 0
    try:
        with tqdm(total=total_frames, desc="💎 AI Upscaling + 1080p Overlays", unit="fr") as pbar:
            while processed_count < total_frames:
                frame = input_queue.get()
                if frame is None:
                    break
                
                # AI Enhancement Pass
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    output, _ = upscaler.enhance(frame, outscale=2)
                
                output_queue.put(output)
                processed_count += 1
                pbar.update(1)
            
            output_queue.put(None) # Signal writer to stop
    finally:
        stop_event.set()
        reader_t.join()
        writer_t.join()
        cap.release()
        if process.stdin:
            process.stdin.close()
        process.wait()

    # 7. AUDIO MUXING
    mux_command = [
        'ffmpeg', '-y',
        '-i', temp_silent_video, 
        '-i', input_path, 
        '-map', '0:v:0', '-map', '1:a:0?', 
        '-c', 'copy', '-shortest', 
        output_path
    ]
    subprocess.run(mux_command, stderr=subprocess.DEVNULL)
    
    if os.path.exists(temp_silent_video): 
        os.remove(temp_silent_video)
    
    elapsed = time.time() - start_time
    print(f"[INFO] Upscale completed in {elapsed:.2f}s")