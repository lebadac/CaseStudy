import os
import logging
from typing import Dict

import cv2
import yaml

from src.utils_io import ensure_dir, valid_image_file, load_image_bgr
from src.resize_methods import resize_with_methods
from src.utils_io import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def save_images_dict(images: Dict[str, "cv2.Mat"], out_dir: str, prefix: str = "") -> None:
    """
    Save a dictionary of images to the specified directory.
    """
    ensure_dir(out_dir)
    for name, img_bgr in images.items():
        filename = f"{prefix}{name}.jpg" if prefix else f"{name}.jpg"
        out_path = os.path.join(out_dir, filename)
        ok = cv2.imwrite(out_path, img_bgr)
        logging.info(f"Saved: {out_path}" if ok else f"Failed to write: {out_path}")


def main():
    """
    Main function to run the batch resize process.
    """
    cfg = load_config("config.yml")

    input_dir = cfg.get("input_dir", "./image")
    out_all = cfg.get("output_all_dir", "./output/all_methods")

    target_w = int(cfg.get("target_w", 1200))
    target_h = int(cfg.get("target_h", 1920))

    if not os.path.exists(input_dir):
        logging.error(f"Input directory not found: {input_dir}")
        return

    processed = 0
    for filename in sorted(os.listdir(input_dir)):
        if not valid_image_file(filename):
            continue

        name, _ = os.path.splitext(filename)
        image_path = os.path.join(input_dir, filename)

        try:
            img_bgr = load_image_bgr(image_path)

            resized = resize_with_methods(img_bgr, target_w, target_h)

            # Save all methods (for fair comparison + evaluation)
            save_images_dict(resized, out_all, prefix=f"{name}_")

            processed += 1

        except Exception as e:
            logging.error(f"Failed to process {filename}: {e}")

    logging.info(f"Batch processing complete. Processed: {processed} images.")


if __name__ == "__main__":
    main()
