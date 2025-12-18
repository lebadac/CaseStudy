from typing import Dict

import cv2

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
