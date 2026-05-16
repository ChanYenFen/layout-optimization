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
    resolve_self_intersections,
    fillet_sharp_corners,
)
from visualization import visualize_multi_contours
import numpy as np
import matplotlib.pyplot as plt


def run_one_correction_step(
    pts,
    img,
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
    verbose_normals=False,
    reference_normals=None,
    it=0,
):
    """
    Run one correction step on a single contour.

    Returns
    -------
    new_pts : (N, 2) ndarray
    indices : list[int]
    core_spans : list[list[int]]
    done : bool   True when no further correction is needed.
    debug_vectors : list[dict]
    inward_normals : dict[int, np.ndarray]
    """
    adjusted_pts, indices, debug_vectors, inward_normals = pull_points(
        pts,
        img,
        min_distance=min_distance,
        search_depth=search_depth,
        increment=increment,
        verbose=verbose,
        verbose_normals=verbose_normals,
        reference_normals=reference_normals,
    )

    if not indices:
        if verbose:
            print(f"[iter {it}] no more pulled points, stop")
        return pts.copy(), [], [], True, debug_vectors, inward_normals

    adjusted_pts, indices = interpolate_modified_spans(
        pts,
        adjusted_pts,
        indices,
        max_gap=interp_gap,
    )

    core_spans = detect_core_spans(indices, len(pts), max_gap=core_gap)

    if not core_spans:
        step_disp = np.linalg.norm(adjusted_pts - pts, axis=1)
        step_max = step_disp.max() if len(step_disp) > 0 else 0.0
        if verbose:
            print(
                f"[iter {it}] no core spans, "
                f"core_points={len(indices)}, "
                f"max_step={step_max:.4f}"
            )
        return adjusted_pts, indices, [], True, debug_vectors, inward_normals

    adjusted_pts = smooth_pull_magnitude_field(
        pts, adjusted_pts, core_spans, passes=mag_smooth_passes
    )

    span_infos = expand_neighborhood(core_spans, len(pts), radius=radius)

    adjusted_pts = apply_decayed_pull(pts, adjusted_pts, span_infos, radius=radius)

    adjusted_pts = smooth_core_displacement(
        pts, adjusted_pts, core_spans, passes=disp_smooth_passes
    )

    adjusted_pts = refit_modified_spans(
        adjusted_pts,
        core_spans,
        anchor_size=refit_anchor_size,
        passes=refit_passes,
        alpha=refit_alpha,
    )

    step_disp = np.linalg.norm(adjusted_pts - pts, axis=1)
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

    done = step_max < 1e-3
    if done and verbose:
        print(f"[iter {it}] converged, stop")

    return adjusted_pts, indices, core_spans, done, debug_vectors, inward_normals


def main(draw_vectors=False):
    max_iter = 4
    params = dict(
        min_distance=30,
        search_depth=20,
        increment=2,
        interp_gap=3,
        core_gap=2,
        radius=50,
        mag_smooth_passes=4,
        disp_smooth_passes=1,
        refit_anchor_size=10,
        refit_passes=4,
        refit_alpha=0.35,
        verbose=True,
        verbose_normals=False,  # set True to log every point's inward vector
    )

    img = load_binary_image("examples/simple_case_3.png")
    contours = extract_contours(img, debug=False)
    print(f"Loaded {len(contours)} contours")

    original_pts_list = [c["points"].astype(float) for c in contours]
    current_pts_list = [pts.copy() for pts in original_pts_list]
    all_debug_vectors = [[] for _ in contours]
    last_indices = [[] for _ in contours]
    last_core_spans = [[] for _ in contours]
    done = [False] * len(contours)
    prev_normals_list = [None] * len(contours)
    reference_normals_list = [None] * len(contours)

    for it in range(max_iter):
        any_active = False
        for k in range(len(contours)):
            if done[k]:
                continue
            new_pts, indices, core_spans, converged, debug_vecs, inward_normals = (
                run_one_correction_step(
                    current_pts_list[k], img, it=it,
                    reference_normals=reference_normals_list[k],
                    **params,
                )
            )

            # Freeze iter-0 normals as the reference for all future iterations.
            if reference_normals_list[k] is None:
                reference_normals_list[k] = inward_normals

            # flip detection: compare inward normals to the previous iteration
            if prev_normals_list[k] is not None:
                prev = prev_normals_list[k]
                flips = [
                    (i, prev[i], inward_normals[i], float(np.dot(inward_normals[i], prev[i])))
                    for i in inward_normals
                    if i in prev and np.dot(inward_normals[i], prev[i]) < 0
                ]
                if flips:
                    print(
                        f"[iter {it}] contour {k}: "
                        f"{len(flips)} normal flip(s)"
                    )
                    for idx, pv, cv, dot in flips:
                        print(
                            f"  point {idx}: "
                            f"prev={np.round(pv, 3)} -> "
                            f"cur={np.round(cv, 3)}  "
                            f"dot={dot:.3f}"
                        )

            prev_normals_list[k] = inward_normals
            all_debug_vectors[k].extend(debug_vecs)
            last_indices[k] = indices
            last_core_spans[k] = core_spans
            current_pts_list[k] = new_pts
            if converged:
                done[k] = True
            else:
                any_active = True

        if not any_active:
            break

    # Post-process: remove self-intersecting loops, then fillet only where
    # intersections were actually resolved (don't touch original geometry).
    for k in range(len(contours)):
        before_n = len(current_pts_list[k])
        current_pts_list[k] = resolve_self_intersections(current_pts_list[k])
        removed = before_n - len(current_pts_list[k])
        if removed > 0:
            print(f"[post] contour {k}: resolved {removed} intersection point(s)")
            before_fillet = len(current_pts_list[k])
            current_pts_list[k] = fillet_sharp_corners(
                current_pts_list[k], min_turn_deg=90.0, radius=5.0, n_arc=4
            )
            added = len(current_pts_list[k]) - before_fillet
            if added > 0:
                print(f"[post] contour {k}: filleted {added} point(s) added for sharp corners")

    results = []
    for k, contour_info in enumerate(contours):
        pts = original_pts_list[k]
        adjusted_pts = current_pts_list[k]
        n_compare = min(len(pts), len(adjusted_pts))
        moved_count = int((np.linalg.norm(adjusted_pts[:n_compare] - pts[:n_compare], axis=1) > 1e-8).sum())

        print(f"\nContour {k}")
        print(f"is_hole: {contour_info.get('is_hole', False)}")
        print(f"depth: {contour_info.get('depth', 0)}")
        print(f"Contour points (original): {len(pts)}, (adjusted): {len(adjusted_pts)}")
        print(f"Core pulled points: {len(last_indices[k])}")
        print(f"Core spans: {len(last_core_spans[k])}")
        print(f"All moved points: {moved_count}")
        print(f"Debug vectors: {len(all_debug_vectors[k])}")

        results.append(
            {
                "id": contour_info.get("id", k),
                "points": pts,
                "adjusted_points": adjusted_pts,
                "core_indices": last_indices[k],
                "core_spans": last_core_spans[k],
                "debug_vectors": all_debug_vectors[k],
                "is_hole": contour_info.get("is_hole", False),
                "depth": contour_info.get("depth", 0),
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
    plt.savefig("/tmp/iterative_result.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main(draw_vectors=True)
