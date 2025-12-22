import torch
import cv2
import numpy as np
import os
import sys
import subprocess
import threading
from queue import Queue
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from tqdm import tqdm

def get_optimal_settings():
    # tile=0 is still best here. Since x2 uses less VRAM, 
    # we can process the whole 480p frame comfortably.
    return 0, True 

def frame_reader(cap, queue, stop_event):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            queue.put(None)
            break
        queue.put(frame)
        while queue.qsize() > 50 and not stop_event.is_set():
            threading.Event().wait(0.01)

def frame_writer(output_queue, ffmpeg_process, stop_event):
    while not stop_event.is_set():
        if output_queue.empty():
            threading.Event().wait(0.01)
            continue
        frame = output_queue.get()
        if frame is None: break
            
        # Final stretch: 960p -> 1080p
        if frame.shape[0] != 1080 or frame.shape[1] != 1920:
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
            
        try:
            ffmpeg_process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            break

def run_upscale(input_path, output_path):
    input_path, output_path = str(input_path), str(output_path)
    tile_size, use_fp16 = get_optimal_settings()
    
    # --- MODEL INIT (Updated for x2) ---
    # RealESRGAN_x2plus uses different architecture params (num_feat=64, but scale=2)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    
    upscaler = RealESRGANer(
        scale=2,
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        model=model,
        tile=tile_size,
        tile_pad=0,
        pre_pad=0,
        half=use_fp16, 
        device='cuda'
    )

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    input_queue = Queue(maxsize=50)
    output_queue = Queue(maxsize=30) 
    stop_event = threading.Event()

    # --- FFMPEG SETUP ---
    command = [
        'ffmpeg', '-y', 
        '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', '1920x1080',
        '-pix_fmt', 'bgr24', '-r', str(fps), '-i', '-',
        '-c:v', 'h264_nvenc', '-preset', 'p2', '-cq', '24', '-pix_fmt', 'yuv420p', 
        output_path
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    reader_t = threading.Thread(target=frame_reader, args=(cap, input_queue, stop_event))
    writer_t = threading.Thread(target=frame_writer, args=(output_queue, process, stop_event))
    reader_t.start()
    writer_t.start()

    print(f"🚀 X2 TURBO ACTIVE: 480p -> 960p (AI) -> 1080p (Lanczos)")

    try:
        with tqdm(total=total_frames, desc="💎 Processing", unit="fr") as pbar:
            while True:
                frame = input_queue.get()
                if frame is None:
                    output_queue.put(None)
                    break
                
                with open(os.devnull, 'w') as f, torch.cuda.amp.autocast(enabled=use_fp16):
                    old_stdout = sys.stdout
                    sys.stdout = f
                    try:
                        # Output will be 960p natively
                        output, _ = upscaler.enhance(frame, outscale=2)
                    finally:
                        sys.stdout = old_stdout
                
                output_queue.put(output)
                pbar.update(1)

    finally:
        stop_event.set()
        reader_t.join()
        writer_t.join()
        cap.release()
        process.stdin.close()
        process.wait()

    print(f"✅ Render Complete: {output_path}")