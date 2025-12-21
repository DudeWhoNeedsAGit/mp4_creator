import torch
import cv2
import numpy as np
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from pathlib import Path

def upscale_video_to_1080p(input_path, output_path):
    """
    Uses the 4070TI Super to upscale video from 480p to 1080p.
    Requires: pip install realesrgan
    """
    print(f"🚀 Starting AI Upscale for: {input_path}")
    
    # 1. Setup Model (RRDBNet is the architecture for Real-ESRGAN)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    
    # Initialize Aligner/Upscaler on CUDA
    upscaler = RealESRGANer(
        scale=4,
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        model=model,
        tile=400, # Large tiles because you have 16GB VRAM
        tile_pad=10,
        pre_pad=0,
        half=True, # Use FP16 for massive speed boost on 4070TI
        device='cuda'
    )

    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define 1080p Output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (1920, 1080))

    while cap.isOpened():
        ret, frame = cap.get()
        if not ret:
            break
        
        # AI Upscale happens here
        output, _ = upscaler.enhance(frame, outscale=2.25) # 480 * 2.25 = 1080
        
        # Resize to exactly 1080p if there's a slight pixel mismatch
        output = cv2.resize(output, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
        
        out.write(output)

    cap.release()
    out.release()
    print(f"✅ Upscale Complete: {output_path}")