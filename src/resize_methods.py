import time
from typing import Dict
import logging 

import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw
from diffusers import AutoPipelineForInpainting
from diffusers.image_processor import VaeImageProcessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

METHODS: Dict[str, int] = {
    "Nearest": cv2.INTER_NEAREST,
    "Linear": cv2.INTER_LINEAR,
    "Cubic": cv2.INTER_CUBIC,
    "Area": cv2.INTER_AREA,
    "Lanczos4": cv2.INTER_LANCZOS4,
}


def resize_with_methods(img_bgr, target_w: int, target_h: int):
    """
    Resize an image using all available interpolation methods.
    """
    out = {}
    for name, interp in METHODS.items():
        out[name] = cv2.resize(img_bgr, (target_w, target_h), interpolation=interp)
    return out

def resize_with_fill(img_bgr, target_w: int, target_h: int):
    """
    Resize an image to (target_w, target_h) and fill the left/right edges using AI inpainting.
    Returns a PIL Image.
    """
    now = time.time()

    # Convert OpenCV BGR to PIL Image
    image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    # Create mask for left and right edges
    mask = Image.new("L", (target_w, target_h), 0)
    draw = ImageDraw.Draw(mask)
    edge_width = int(target_w * 0.048)  # ~58px for 1200px width
    draw.rectangle([0, 0, edge_width, target_h], fill=255)
    draw.rectangle([target_w - edge_width, 0, target_w, target_h], fill=255)

    # Resize image using VaeImageProcessor for best quality
    processor = VaeImageProcessor()
    resized_image = processor._resize_and_fill(image, width=target_w, height=target_h)

    # Load inpainting pipeline
    pipeline = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16
    ).to("cuda")

    prompt = (
        "Extend the original image by filling only the masked left and right edges with visually consistent, photorealistic background. "
        "Do not alter the main subject or unmasked regions. "
        "Ensure seamless blending, high detail, and realistic textures. "
        "No artifacts, no repetition, and no changes to the central content."
    )

    result = pipeline(
        prompt=prompt,
        image=resized_image,
        mask_image=mask,
        height=target_h,
        width=target_w,
        strength=0.99
    ).images[0]
    end = time.time()
    logging.info(f"Resized with AI fill in {end - now:.2f}s")
    return result