import os
import logging
from typing import Optional
import time 

import cv2

from src.utils_io import ensure_dir, valid_image_file, load_image_bgr
from src.resize_methods import METHODS, resize_with_fill
from src.utils_io import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    """
    Main function to run the batch resize process.
    """

    cfg = load_config("config.yml")

    input_dir = cfg.get("input_dir", "./image")
    out_best = cfg.get("output_best_dir", "./output/best")

    target_w = int(cfg.get("target_w", 1200))
    target_h = int(cfg.get("target_h", 1920))

    best_method = cfg.get("best_method")

    if not os.path.exists(input_dir):
        logging.error(f"Input directory not found: {input_dir}")
        return

    ensure_dir(out_best)

    processed = 0
    for filename in sorted(os.listdir(input_dir)):
        if not valid_image_file(filename):
            continue

        name, _ = os.path.splitext(filename)
        image_path = os.path.join(input_dir, filename)

        try:
            img_bgr = load_image_bgr(image_path)

            if best_method == "AI_Fill":
                # Resize bằng AI fill
                result_img = resize_with_fill(img_bgr, target_w, target_h)
                out_path = os.path.join(out_best, f"{name}_AI_Fill.jpg")
                result_img.save(out_path)
                ok = True
            elif best_method in METHODS:
                # Resize truyền thống
                now = time.time()
                resized = cv2.resize(
                    img_bgr,
                    (target_w, target_h),
                    interpolation=METHODS[best_method],
                )
                end = time.time()
                logging.info(f"Resized {filename} with {best_method} in {end - now:.2f}s")
                out_path = os.path.join(out_best, f"{name}_{best_method}.jpg")
                ok = cv2.imwrite(out_path, resized)
            else:
                logging.error(
                    f"best_method='{best_method}' not in available methods: {list(METHODS.keys()) + ['AI_Fill']}"
                )
                continue

            logging.info(f"Saved: {out_path}" if ok else f"Failed to write: {out_path}")
            processed += 1

        except Exception as e:
            logging.error(f"Failed to process {filename}: {e}")

    logging.info(
        f"Batch BEST processing complete. Method='{best_method}'. Images processed: {processed}"
    )


if __name__ == "__main__":
    main()
