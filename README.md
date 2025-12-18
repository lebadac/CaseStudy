# Case Study: Image Interpolation

This project provides tools for resizing and evaluating images using various OpenCV interpolation methods.

## Prerequisites

- Python 3.10+
- [Conda](https://docs.conda.io/en/latest/) (optional but recommended)

## Setup

1.  **Create Environment** (Optional):
    ```bash
    conda create -n case_study python=3.10
    conda activate case_study
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Project Structure

The project logic is organized as follows:

-   **`main.py`**: Entry point for running the batch "Best Method" processing pipeline.
-   **`config.yml`**: Configuration file for paths, target dimensions, and the preferred method.
-   **`src/`**:
    -   **`src/batch_resize.py`**: Handles batch resizing logic.
    -   **`src/evaluate_methods.py`**: Benchmarking logic using PSNR and SSIM metrics.
    -   **`src/resize_methods.py`**: Defines the OpenCV interpolation methods.
    -   **`src/utils_io.py`**: Utility functions for file I/O and config loading.

## Configuration

Control the behavior via `config.yml`:

```yaml
best_method: "Lanczos4"
target_w: 1200
target_h: 1920
input_dir: "./image"
output_best_dir: "./output/best"
```

## Usage

Follow these steps for the complete workflow:

1.  **Step 1: Prepare Images**
    Place your source images in the `./image/` folder.

2.  **Step 2: Resize with all Methods**
    To generate all 5 interpolation variations for comparison:
    ```bash
    python -m src.batch_resize
    ```

3.  **Step 3: Evaluate**
    To see objective quality metrics (PSNR/SSIM) for all available methods:
    ```bash
    python -m src.evaluate_methods
    ```

4.  **Step 4: Edit Config**
    Based on the evaluation results, update `best_method` in `config.yml` (e.g., `best_method: "Lanczos4"`).

5.  **Step 5: Output Run with main.py**
    To process and save images using your selected "best" method:
    ```bash
    python -m main
    ```

## Output

Results are saved to the directory specified in `config.yml` (default: `./output/best/`).
If you run an "all methods" process (via evaluation or internal scripts), variations are typically stored in `./output/all_methods/`.
# CaseStudy
