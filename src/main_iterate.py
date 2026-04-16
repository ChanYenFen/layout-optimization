from cv.rasterize import load_binary_image, extract_contour
from geometry.correction import (
    pull_points,
    interpolate_modified_spans,
    detect_core_spans,
    smooth_pull_magnitude_field,
    expand_neighborhood,
    apply_decayed_pull,
    refit_modified_spans,
)
from visualization import visualize_adjusted_points
import numpy as np


def main():
    img = load_binary_image("examples/simple_case.png")
    pts = extract_contour(img)

    current_pts = pts.copy()

    max_iter = 5
    min_distance = 15
    increment = 3
    interp_gap = 10
    core_gap = 2
    radius = 30

    for it in range(max_iter):
        adjusted_pts, indices = pull_points(
            current_pts,
            img,
            min_distance=min_distance,
            increment=increment,
        )

        if len(indices) == 0:
            print(f"[iter {it}] no more pulled points, stop")
            break

        adjusted_pts, indices = interpolate_modified_spans(
            current_pts,
            adjusted_pts,
            indices,
            max_gap=interp_gap,
        )

        core_spans = detect_core_spans(indices, len(current_pts), max_gap=core_gap)

        adjusted_pts = smooth_pull_magnitude_field(
            current_pts,
            adjusted_pts,
            core_spans,
            passes=4,
        )

        span_infos = expand_neighborhood(
            core_spans,
            len(current_pts),
            radius=radius,
        )

        adjusted_pts = apply_decayed_pull(
            current_pts,
            adjusted_pts,
            span_infos,
            radius=radius,
        )

        adjusted_pts = refit_modified_spans(
            adjusted_pts,
            core_spans,
            anchor_size=4,
            passes=4,
            alpha=0.35,
        )

        step_disp = np.linalg.norm(adjusted_pts - current_pts, axis=1)
        step_max = step_disp.max()
        moved_count = int((step_disp > 1e-8).sum())

        print(
            f"[iter {it}] "
            f"core_points={len(indices)}, "
            f"core_spans={len(core_spans)}, "
            f"moved={moved_count}, "
            f"max_step={step_max:.4f}"
        )

        current_pts = adjusted_pts

        if step_max < 1e-3:
            print(f"[iter {it}] converged, stop")
            break

    moved_mask = np.linalg.norm(current_pts - pts, axis=1) > 1e-8
    moved_indices = np.where(moved_mask)[0].tolist()

    visualize_adjusted_points(
        pts,
        current_pts,
        moved_indices,
        title="Original vs Iterative Corrected"
    )

    print(f"Loaded {len(pts)} contour points")
    print(f"Final moved points: {len(moved_indices)}")


if __name__ == "__main__":
    main()