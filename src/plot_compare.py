import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.resize_methods import resize_with_fill

def plot_comparison(image_path, target_w, target_h, save_path):
    # Load original image (BGR)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Cannot load image: {image_path}")
        return

    # Convert to RGB for matplotlib
    orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Resize with INTER_LANCZOS4
    lanczos4 = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    lanczos4_rgb = cv2.cvtColor(lanczos4, cv2.COLOR_BGR2RGB)

    # Resize with AI fill (PIL Image to np.array)
    ai_filled_pil = resize_with_fill(img_bgr, target_w, target_h)
    ai_filled_rgb = np.array(ai_filled_pil)

    # Plot
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(orig_rgb)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(lanczos4_rgb)
    plt.title("INTER_LANCZOS4")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(ai_filled_rgb)
    plt.title("AI Fill")
    plt.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Comparison plot saved to {save_path}")
    

if __name__ == "__main__":
    # Example usage
    image_path = "./image/2.png" 
    target_w = 1200
    target_h = 1920
    save_path = "./image_readme/comparison_2.png"
    plot_comparison(image_path, target_w, target_h, save_path)