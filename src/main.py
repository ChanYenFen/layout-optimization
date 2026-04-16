from cv.rasterize import load_binary_image, extract_contour
from geometry.correction import (
    pull_points,
    interpolate_modified_spans,
    detect_core_spans,
    smooth_pull_magnitude_field,
    smooth_core_displacement,
    refit_modified_spans,
    expand_neighborhood,
    apply_decayed_pull
)
from visualization import visualize_adjusted_points
import numpy as np


def main():
    img = load_binary_image("examples/simple_case.png")
    pts = extract_contour(img)

    # 1. core pull
    adjusted_pts, indices = pull_points(
        pts,
        img,
        min_distance=15,
        increment=5,
    )

    # 2. fill small gaps inside modified regions
    adjusted_pts, indices = interpolate_modified_spans(
        pts,
        adjusted_pts,
        indices,
        max_gap=10,
    )

    # 3. detect core spans
    core_spans = detect_core_spans(indices, len(pts), max_gap=2)

    # 4. smooth pull magnitude first
    adjusted_pts = smooth_pull_magnitude_field(
        pts,
        adjusted_pts,
        core_spans,
        passes=4,
    )

    # 5. expand neighborhood
    span_infos = expand_neighborhood(
        core_spans,
        len(pts),
        radius=10,
    )

    # 6. apply decayed pull
    adjusted_pts = apply_decayed_pull(
        pts,
        adjusted_pts,
        span_infos,
        radius=10,
    )

    # 7. optional weak vector smoothing
    adjusted_pts = smooth_core_displacement(
        pts,
        adjusted_pts,
        core_spans,
        passes=1,
    )

    # 8. local refit
    adjusted_pts = refit_modified_spans(
        adjusted_pts,
        core_spans,
        anchor_size=4,
        passes=4,
        alpha=0.35,
    )

    moved_mask = np.linalg.norm(adjusted_pts - pts, axis=1) > 1e-8
    moved_indices = np.where(moved_mask)[0].tolist()

    visualize_adjusted_points(
        pts,
        adjusted_pts,
        moved_indices,
        title="Original vs Corrected"
    )

    print(f"Loaded {len(pts)} contour points")
    print(f"Core pulled points: {len(indices)}")
    print(f"Core spans: {len(core_spans)}")
    print(f"All moved points: {len(moved_indices)}")

if __name__ == "__main__":
    main()