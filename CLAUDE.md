# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Python project for algorithmic contour correction on binary images. It detects narrow local regions in 2D contours (extracted from raster images) and selectively corrects them while preserving boundary continuity.

## Running the pipeline

All scripts must be run from the `src/` directory so that relative imports and image paths resolve correctly:

```bash
cd src

# Single-pass pipeline on a test image:
python main.py

# Iterative pipeline (preferred, converges toward stable correction):
python main_iterate.py
```

The `examples/` folder at project root contains the test PNGs. `main.py` uses `simple_case_2.png`; `main_iterate.py` uses `simple_case_4.png`. To try a different image, change the `load_binary_image(...)` call in `main()`.

## Python environment

The project uses Python 3.9.13 with a venv at `/Users/chanyenfen/cyf/`. Key dependencies: `numpy`, `opencv-python`, `matplotlib`. No `requirements.txt` is present; install with:

```bash
pip install numpy opencv-python matplotlib
```

## Architecture

The pipeline has three layers:

### 1. Image / contour I/O (`src/cv/rasterize.py`)
- `load_binary_image(path)` — loads a PNG as a binary (0/255) ndarray.
- `extract_contours(binary_img)` — wraps `cv2.findContours` (RETR_TREE) and returns a list of dicts, each with `points` (N×2 ndarray), `depth`, `is_hole`, `parent`, `area`. Contours are sorted by `(depth, -area)`.

### 2. Correction pipeline (`src/geometry/correction/`)
Each step operates on `(N, 2) float64` point arrays and returns a modified copy. The steps in order:

| Function | Module | Purpose |
|---|---|---|
| `pull_points` | `pull.py` | Detect thickness violations; displace points outward along inward normal |
| `interpolate_modified_spans` | `span.py` | Fill small gaps between pulled points via displacement interpolation |
| `detect_core_spans` | `span.py` | Cluster pulled indices into contiguous spans |
| `smooth_pull_magnitude_field` | `smooth.py` | Smooth pull *magnitude* within each span (preserves per-point direction) |
| `expand_neighborhood` | `span.py` | Extend each span left/right by `radius` indices |
| `apply_decayed_pull` | `decay.py` | Apply linearly-decayed displacement to neighbors outside core spans |
| `smooth_core_displacement` | `smooth.py` | Smooth full displacement vector within each core span |
| `refit_modified_spans` | `refit.py` | Local Laplacian refit anchored outside each span |

The `__init__.py` re-exports all of these for clean imports in `main.py` / `main_iterate.py`.

### 3. Entry points
- `main.py` — single-pass: runs the pipeline once per contour.
- `main_iterate.py` — iterative: repeats the pipeline up to `max_iter` times per contour, stopping early when `max_step < 1e-3` or no violations remain. Preferred for cases where one pass leaves residual violations.

### Visualization (`src/visualization.py`)
- `visualize_multi_contours(img, results, draw_vectors=False)` — overlays original (dashed) and corrected (solid) contours for all contours in one plot; optionally draws per-point pull vectors. All plots use inverted Y axis (image coordinate convention).

## Key design notes

- All contour points are stored as `(N, 2) float64` ndarrays in image-space coordinates (x right, y down).
- The pipeline never modifies arrays in place; each stage returns a new ndarray.
- `correction_legacy.py` in `src/geometry/` is an old monolithic version of the correction logic kept for reference; active code lives in `src/geometry/correction/`.
- `debug_width.py` and `tempCodeRunnerFile.py` in `src/` are scratch/debug scripts, not part of the pipeline.
- `archive/` and `experiments/` are intentionally empty placeholder directories.
- The README project structure diagram is aspirational (describes a planned layout); the actual layout is as described above.
