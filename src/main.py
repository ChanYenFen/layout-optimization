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


def run_single_contour_pipeline(pts, img):
    adjusted_pts, indices, debug_vectors = pull_points(
        pts,
        img,
        min_distance=50,
        increment=5,
        verbose=True,
    )

    if not indices:
        return pts.copy(), [], [], [], debug_vectors

    adjusted_pts, indices = interpolate_modified_spans(
        pts,
        adjusted_pts,
        indices,
        max_gap=10,
    )

    core_spans = detect_core_spans(indices, len(pts), max_gap=2)

    if not core_spans:
        moved_mask = np.linalg.norm(adjusted_pts - pts, axis=1) > 1e-8
        moved_indices = np.where(moved_mask)[0].tolist()
        return adjusted_pts, indices, [], moved_indices, debug_vectors

    adjusted_pts = smooth_pull_magnitude_field(
        pts,
        adjusted_pts,
        core_spans,
        passes=4,
    )

    span_infos = expand_neighborhood(
        core_spans,
        len(pts),
        radius=10,
    )

    adjusted_pts = apply_decayed_pull(
        pts,
        adjusted_pts,
        span_infos,
        radius=10,
    )

    adjusted_pts = smooth_core_displacement(
        pts,
        adjusted_pts,
        core_spans,
        passes=1,
    )

    adjusted_pts = refit_modified_spans(
        adjusted_pts,
        core_spans,
        anchor_size=4,
        passes=4,
        alpha=0.35,
    )

    moved_mask = np.linalg.norm(adjusted_pts - pts, axis=1) > 1e-8
    moved_indices = np.where(moved_mask)[0].tolist()

    return adjusted_pts, indices, core_spans, moved_indices, debug_vectors


def main(draw_vectors=False):
    img = load_binary_image("examples/simple_case_3.png")
    contours = extract_contours(img, debug=False)

    print(f"Loaded {len(contours)} contours")

    results = []

    for k, contour_info in enumerate(contours):
        pts = contour_info["points"]
        is_hole = contour_info.get("is_hole", False)
        depth = contour_info.get("depth", 0)

        adjusted_pts, indices, core_spans, moved_indices, debug_vectors = (
            run_single_contour_pipeline(pts, img)
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
        title="All contours: original vs corrected",
        draw_vectors=draw_vectors,
    )


if __name__ == "__main__":
    main(draw_vectors=True)