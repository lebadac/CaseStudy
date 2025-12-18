import os
import csv
import math
import logging
from typing import Dict, Optional, Tuple, List

import cv2
import yaml

from src.utils_io import ensure_dir, valid_image_file, load_image_bgr
from src.resize_methods import METHODS
from src.utils_io import load_config
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Optional SSIM
try:
    from skimage.metrics import structural_similarity as sk_ssim  # type: ignore
    HAS_SSIM = True
except Exception:
    HAS_SSIM = False

def to_gray(img_bgr):
    """
    Convert BGR image to grayscale.
    """
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def compute_psnr(original_bgr, recon_bgr) -> float:
    """
    Compute PSNR between two images.
    """
    return float(cv2.PSNR(original_bgr, recon_bgr))


def compute_ssim_gray(original_gray, recon_gray) -> Optional[float]:
    """
    Compute SSIM between two grayscale images.
    """
    if not HAS_SSIM:
        return None
    return float(sk_ssim(original_gray, recon_gray, data_range=255))


def roundtrip_resize(
    img_bgr,
    upscale_size: Tuple[int, int],
    upscale_interp: int,
    downscale_interp: int = cv2.INTER_AREA,
):
    """
    Perform roundtrip resize.
    """
    up_w, up_h = upscale_size
    h0, w0 = img_bgr.shape[:2]

    up = cv2.resize(img_bgr, (up_w, up_h), interpolation=upscale_interp)
    back = cv2.resize(up, (w0, h0), interpolation=downscale_interp)
    return up, back


def write_csv(path: str, header: List[str], rows: List[List[str]]) -> None:
    """
    Write CSV file.
    """
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    logging.info(f"Wrote CSV: {path}")


def mean_std(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute mean and standard deviation.
    """
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return mean, math.sqrt(var)


def main():
    """
    Main function.
    """
    cfg = load_config("config.yml")

    input_dir = cfg.get("input_dir", "./image")
    out_dir = cfg.get("output_eval_dir", "./output_eval")
    save_roundtrip = bool(cfg.get("save_roundtrip_images", True))

    target_w = int(cfg.get("target_w", 1200))
    target_h = int(cfg.get("target_h", 1920))
    upscale_size = (target_w, target_h)

    if not os.path.exists(input_dir):
        logging.error(f"Input directory not found: {input_dir}")
        return

    if not HAS_SSIM:
        logging.warning("SSIM is disabled (scikit-image not installed). Only PSNR will be computed.")
        logging.warning("Enable SSIM with: pip install scikit-image")

    roundtrip_dir = os.path.join(out_dir, "roundtrip")
    per_csv = os.path.join(out_dir, "metrics_per_image.csv")
    sum_csv = os.path.join(out_dir, "metrics_summary.csv")

    # records: list of dict-like rows
    records = []

    for filename in sorted(os.listdir(input_dir)):
        if not valid_image_file(filename):
            continue

        path = os.path.join(input_dir, filename)
        name, _ = os.path.splitext(filename)

        try:
            original = load_image_bgr(path)

            for method_name, interp in METHODS.items():
                _, back = roundtrip_resize(original, upscale_size, interp, downscale_interp=cv2.INTER_AREA)

                psnr = compute_psnr(original, back)
                ssim_val = compute_ssim_gray(to_gray(original), to_gray(back)) if HAS_SSIM else None

                records.append({
                    "image": filename,
                    "method": method_name,
                    "psnr": psnr,
                    "ssim": ssim_val,
                })

                if save_roundtrip:
                    ensure_dir(roundtrip_dir)
                    out_path = os.path.join(roundtrip_dir, f"{name}_{method_name}_roundtrip.jpg")
                    cv2.imwrite(out_path, back)

            logging.info(f"Evaluated: {filename}")

        except Exception as e:
            logging.error(f"Failed to evaluate {filename}: {e}")

    # Per-image CSV
    per_rows = []
    for r in records:
        per_rows.append([
            r["image"],
            r["method"],
            f"{r['psnr']:.4f}",
            "" if r["ssim"] is None else f"{r['ssim']:.6f}",
        ])

    write_csv(
        per_csv,
        header=["image", "method", "psnr_roundtrip", "ssim_roundtrip"],
        rows=per_rows,
    )

    # Summary per method
    summary_rows = []
    summary_for_sort = []
    for method_name in METHODS.keys():
        psnrs = [r["psnr"] for r in records if r["method"] == method_name]
        ssims = [r["ssim"] for r in records if r["method"] == method_name and r["ssim"] is not None]

        psnr_mean, psnr_std = mean_std(psnrs)
        ssim_mean, ssim_std = mean_std(ssims)

        summary_rows.append([
            method_name,
            str(len(psnrs)),
            "" if psnr_mean is None else f"{psnr_mean:.4f}",
            "" if psnr_std is None else f"{psnr_std:.4f}",
            "" if ssim_mean is None else f"{ssim_mean:.6f}",
            "" if ssim_std is None else f"{ssim_std:.6f}",
        ])

        summary_for_sort.append((method_name, psnr_mean if psnr_mean is not None else -1.0, ssim_mean))

    write_csv(
        sum_csv,
        header=["method", "count", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std"],
        rows=summary_rows,
    )

    # Print ranking
    summary_for_sort.sort(key=lambda x: x[1], reverse=True)
    logging.info("Ranking (higher PSNR mean is better):")
    for i, (m, psnr_m, ssim_m) in enumerate(summary_for_sort, start=1):
        logging.info(f"{i}. {m} | PSNR mean={psnr_m:.4f} | SSIM mean={'N/A' if ssim_m is None else f'{ssim_m:.6f}'}")

    logging.info("Done.")


if __name__ == "__main__":
    main()
