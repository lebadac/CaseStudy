import os
import logging
from typing import Set

import cv2
import yaml

VALID_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp"}


def ensure_dir(path: str) -> None:
    """
    Ensure the directory exists, creating it if necessary.
    """
    os.makedirs(path, exist_ok=True)


def valid_image_file(filename: str) -> bool:
    """
    Check if the filename has a valid image extension.
    """
    _, ext = os.path.splitext(filename)
    return ext.lower() in VALID_EXTENSIONS


def load_image_bgr(path: str):
    """
    Load an image in BGR format.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape) == 3 else 1
    logging.info(f"Loaded: {path} | (H,W,C)=({h},{w},{c})")
    return img


def load_config(path: str = "config.yml") -> dict:
    """
    Load a YAML configuration file.
    """
    if not os.path.exists(path):
        logging.warning(f"Config file {path} not found. Using defaults.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
