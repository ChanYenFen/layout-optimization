from cv.rasterize import load_binary_image, extract_contours
from geometry.correction import (
    pull_points,
    interpolate_modified_spans,
    detect_core_spans,
    smooth_pull_magnitude_field,
    smooth_core_displacement,
    refit_modified_spans,
    expand_neighborhood,
    apply_decayed_pull,
)
from visualization import visualize_multi_contours
import numpy as np


def run_iterative_contour_pipeline(
    pts,
    img,
    max_iter=5,
    min_distance=50,
    search_depth=50,
    increment=3,
    interp_gap=10,
    core_gap=2,
    radius=10,
    mag_smooth_passes=4,
    disp_smooth_passes=1,
    refit_anchor_size=20,
    refit_passes=4,
    refit_alpha=0.35,
    verbose=True,
):
    current_pts = pts.copy()
    all_debug_vectors = []
    last_indices = []
    last_core_spans = []

    for it in range(max_iter):
        adjusted_pts, indices, debug_vectors = pull_points(
            current_pts,
            img,
            min_distance=min_distance,
            search_depth=search_depth,
            increment=increment,
            verbose=verbose,
        )
        all_debug_vectors.extend(debug_vectors)

        if not indices:
            if verbose:
                print(f"[iter {it}] no more pulled points, stop")
            break

        adjusted_pts, indices = interpolate_modified_spans(
            current_pts,
            adjusted_pts,
            indices,
            max_gap=interp_gap,
        )

        core_spans = detect_core_spans(
            indices,
            len(current_pts),
            max_gap=core_gap,
        )

        if not core_spans:
            step_disp = np.linalg.norm(adjusted_pts - current_pts, axis=1)
            step_max = step_disp.max() if len(step_disp) > 0 else 0.0
            current_pts = adjusted_pts
            last_indices = indices
            last_core_spans = []
            if verbose:
                print(
                    f"[iter {it}] no core spans, "
                    f"core_points={len(indices)}, "
                    f"max_step={step_max:.4f}"
                )
            break

        adjusted_pts = smooth_pull_magnitude_field(
            current_pts,
            adjusted_pts,
            core_spans,
            passes=mag_smooth_passes,
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

        adjusted_pts = smooth_core_displacement(
            current_pts,
            adjusted_pts,
            core_spans,
            passes=disp_smooth_passes,
        )

        adjusted_pts = refit_modified_spans(
            adjusted_pts,
            core_spans,
            anchor_size=refit_anchor_size,
            passes=refit_passes,
            alpha=refit_alpha,
        )

        step_disp = np.linalg.norm(adjusted_pts - current_pts, axis=1)
        step_max = step_disp.max() if len(step_disp) > 0 else 0.0
        moved_count = int((step_disp > 1e-8).sum())

        if verbose:
            print(
                f"[iter {it}] "
                f"core_points={len(indices)}, "
                f"core_spans={len(core_spans)}, "
                f"moved={moved_count}, "
                f"max_step={step_max:.4f}"
            )

        current_pts = adjusted_pts
        last_indices = indices
        last_core_spans = core_spans

        if step_max < 1e-3:
            if verbose:
                print(f"[iter {it}] converged, stop")
            break

    moved_mask = np.linalg.norm(current_pts - pts, axis=1) > 1e-8
    moved_indices = np.where(moved_mask)[0].tolist()

    return current_pts, last_indices, last_core_spans, moved_indices, all_debug_vectors


def main(draw_vectors=False):
    img = load_binary_image("examples/simple_case_2.png")
    contours = extract_contours(img, debug=False)

    print(f"Loaded {len(contours)} contours")

    results = []

    for k, contour_info in enumerate(contours):
        pts = contour_info["points"]
        is_hole = contour_info.get("is_hole", False)
        depth = contour_info.get("depth", 0)

        adjusted_pts, indices, core_spans, moved_indices, debug_vectors = (
            run_iterative_contour_pipeline(
                pts,
                img,
                max_iter=2,
                min_distance=30,
                search_depth=50,
                increment=3,   # iterative: smaller step than main.py
                interp_gap=10,
                core_gap=2,
                radius=50,
                mag_smooth_passes=4,
                disp_smooth_passes=1,
                refit_anchor_size=10,
                refit_passes=4,
                verbose=True,
            )
        )

        print(f"\nContour {k}")
        print(f"is_hole: {is_hole}")
        print(f"depth: {depth}")
        print(f"Contour points: {len(pts)}")
        print(f"Core pulled points: {len(indices)}")
        print(f"Core spans: {len(core_spans)}")
        print(f"All moved points: {len(moved_indices)}")
        print(f"Debug vectors: {len(debug_vectors)}")

        results.append(
            {
                "id": contour_info.get("id", k),
                "points": pts,
                "adjusted_points": adjusted_pts,
                "moved_indices": moved_indices,
                "core_indices": indices,
                "core_spans": core_spans,
                "debug_vectors": debug_vectors,
                "is_hole": is_hole,
                "depth": depth,
                "parent": contour_info.get("parent", -1),
                "area": contour_info.get("area", 0.0),
            }
        )

    visualize_multi_contours(
        img,
        results,
        title="All contours: iterative corrected",
        draw_vectors=draw_vectors,
    )


if __name__ == "__main__":
    main(draw_vectors=True)