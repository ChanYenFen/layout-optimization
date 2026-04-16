"""
geometry/correction.py

Constraint-aware contour correction utilities.

This version keeps the original `pull_points` entry point and adds
`pull_points_with_falloff`, which propagates pull displacement to nearby
points with linear falloff.

Assumptions
-----------
- `points` is an (N, 2) numpy array.
- `img` is a 2D binary numpy array.
- Foreground / valid region is non-zero.
- Contour is a closed loop.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Basic geometry helpers
# -----------------------------------------------------------------------------

def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return normalized vector."""
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v, dtype=float)
    return v / n


def compute_tangent(points: np.ndarray, i: int) -> np.ndarray:
    """Estimate tangent from previous and next contour points."""
    n = len(points)
    prev_pt = points[(i - 1) % n].astype(float)
    next_pt = points[(i + 1) % n].astype(float)
    return _normalize(next_pt - prev_pt)


def compute_normal(points: np.ndarray, i: int) -> np.ndarray:
    """Return one candidate 2D normal from tangent."""
    t = compute_tangent(points, i)
    # rotate 90 deg: (x, y) -> (-y, x)
    return np.array([-t[1], t[0]], dtype=float)


def _sample_binary(img: np.ndarray, pt: np.ndarray) -> int:
    """Nearest-neighbor sample from binary image. Out of bounds -> 0."""
    x = int(round(float(pt[0])))
    y = int(round(float(pt[1])))
    h, w = img.shape[:2]
    if x < 0 or x >= w or y < 0 or y >= h:
        return 0
    return int(img[y, x])


def get_inward_normal(points: np.ndarray, i: int, img: np.ndarray, probe: float = 1.5) -> np.ndarray:
    """
    Pick inward normal by probing both normal directions.

    Heuristic:
    - If one side lands on foreground and the other does not, use foreground side.
    - Otherwise return the base normal.
    """
    base_n = compute_normal(points, i)
    if np.allclose(base_n, 0.0):
        return base_n

    p = points[i].astype(float)
    a = p + base_n * probe
    b = p - base_n * probe

    va = _sample_binary(img, a)
    vb = _sample_binary(img, b)

    if va > 0 and vb == 0:
        return base_n
    if vb > 0 and va == 0:
        return -base_n
    return base_n


# -----------------------------------------------------------------------------
# Narrow-region probing
# -----------------------------------------------------------------------------

def _ray_hit_distance(
    point: np.ndarray,
    direction: np.ndarray,
    img: np.ndarray,
    max_distance: float,
    step: float = 1.0,
) -> float | None:
    """
    March from point along direction until leaving foreground.

    Returns the traveled distance when the sampled pixel becomes background.
    Returns None if no hit occurs within max_distance.
    """
    d = _normalize(direction)
    if np.allclose(d, 0.0):
        return None

    dist = step
    while dist <= max_distance:
        p = point + d * dist
        if _sample_binary(img, p) == 0:
            return dist
        dist += step
    return None


# -----------------------------------------------------------------------------
# Original pull_points (kept simple and explicit)
# -----------------------------------------------------------------------------

def pull_points(
    points: np.ndarray,
    binary_img: np.ndarray,
    min_distance: float = 5,
    increment: float = 5,
) -> tuple[np.ndarray, list[int]]:
    """
    Pull boundary points outward when the opposite side of the shape
    is found too close along the inward normal direction.
    """
    adjusted = points.astype(np.float64).copy()
    indices_to_adjust: list[int] = []

    h, w = binary_img.shape
    max_steps = int(np.ceil(min_distance))

    for i in range(len(points)):
        inward_normal = get_inward_normal(points, i, binary_img)

        print(f"[{i}] inward_normal = {inward_normal}")

        if np.linalg.norm(inward_normal) == 0:
            continue

        outward_normal = -inward_normal
        p = points[i].astype(np.float64)

        for d in range(1, max_steps + 1):
            test_pt = p + inward_normal * d
            x, y = int(round(test_pt[0])), int(round(test_pt[1]))

            if x < 0 or x >= w or y < 0 or y >= h:
                break

            if binary_img[y, x] == 0:
                continue

            opposite_found = False

            for dd in range(1, max_steps + 1):
                probe_pt = test_pt + inward_normal * dd
                px, py = int(round(probe_pt[0])), int(round(probe_pt[1]))

                if px < 0 or px >= w or py < 0 or py >= h:
                    break

                if binary_img[py, px] == 0:
                    distance = d
                    print(f"[{i}] distance = {distance}")

                    if distance < min_distance:
                        violation = max(min_distance - distance, 0.0)
                        pull_dist = min(np.sqrt(violation) * 1.2, increment)
                        # pull_dist = min((min_distance - distance) / 2.0, increment)
                        adjusted[i] = p + outward_normal * pull_dist
                        indices_to_adjust.append(i)

                        print(
                            f"PULL [{i}] distance={distance}, "
                            f"pull_dist={pull_dist}, "
                            f"from={p} to={adjusted[i]}"
                        )

                    opposite_found = True
                    break

            if opposite_found:
                break

    return adjusted, indices_to_adjust

def interpolate_modified_spans(original_pts, adjusted_pts, indices, max_gap=2):
    """
    Fill small unmoved gaps inside a modified region by interpolating displacement.

    Parameters
    ----------
    original_pts : (N, 2) ndarray
    adjusted_pts : (N, 2) ndarray
    indices : list[int]
        Indices directly modified by pull_points.
    max_gap : int
        If two modified indices are close enough, treat them as the same region.

    Returns
    -------
    new_pts : (N, 2) ndarray
    new_indices : list[int]
        Updated indices after filling gaps.
    """
    if not indices:
        return adjusted_pts.copy(), []

    new_pts = adjusted_pts.astype(float).copy()
    disp = new_pts - original_pts
    idx = sorted(set(indices))

    # merge nearby modified points into one region
    segments = [[idx[0]]]
    for cur in idx[1:]:
        if cur - segments[-1][-1] <= max_gap + 1:
            segments[-1].append(cur)
        else:
            segments.append([cur])

    filled = set(idx)

    for seg in segments:
        start = seg[0]
        end = seg[-1]

        # walk through consecutive modified points in this segment
        for a, b in zip(seg[:-1], seg[1:]):
            gap = b - a
            if gap <= 1:
                continue

            # linear interpolation of displacement across the gap
            da = disp[a]
            db = disp[b]

            for k in range(1, gap):
                t = k / float(gap)
                i = a + k
                interp_d = (1 - t) * da + t * db
                new_pts[i] = original_pts[i] + interp_d
                filled.add(i)

    return new_pts, sorted(filled)

def detect_core_spans(indices, n, max_gap=2):
    """
    Merge modified indices into core spans.
    Small gaps inside a span are filled.

    Returns
    -------
    spans : list[list[int]]
        Each span is a contiguous index list.
    """
    if not indices:
        return []

    idx = sorted(set(i % n for i in indices))
    spans = [[idx[0]]]

    for cur in idx[1:]:
        last = spans[-1][-1]
        gap = cur - last - 1

        if gap <= max_gap:
            spans[-1].extend(range(last + 1, cur + 1))
        else:
            spans.append([cur])

    return spans


def expand_neighborhood(spans, n, radius=4):
    """
    Expand each core span to left/right neighbors.

    Returns
    -------
    span_infos : list[dict]
        {
            "core": [...],
            "left": [(idx, dist), ...],
            "right": [(idx, dist), ...]
        }
    """
    span_infos = []

    for span in spans:
        left_edge = span[0]
        right_edge = span[-1]

        left_neighbors = [((left_edge - d) % n, d) for d in range(1, radius + 1)]
        right_neighbors = [((right_edge + d) % n, d) for d in range(1, radius + 1)]

        span_infos.append({
            "core": span,
            "left": left_neighbors,
            "right": right_neighbors,
        })

    return span_infos


# def apply_decayed_pull(original_pts, adjusted_pts, span_infos, radius=4):
#     """
#     Apply decayed pull to neighbors around each core span.

#     Core points keep their original pull displacement.
#     Left/right neighbors inherit weaker displacement from the span edges.
#     """
#     original_pts = original_pts.astype(float)
#     adjusted_pts = adjusted_pts.astype(float)

#     disp = adjusted_pts - original_pts
#     n = len(original_pts)

#     out_disp = disp.copy()
#     accum = np.zeros_like(disp)
#     weight_sum = np.zeros((n, 1), dtype=float)

#     core_set = set()
#     for info in span_infos:
#         core_set.update(info["core"])

#     for info in span_infos:
#         core = info["core"]
#         mean_vec = np.mean(disp[core], axis=0)

#         for idx, dist in info["left"]:
#             if idx in core_set:
#                 continue
#             w = 1.0 - dist / float(radius + 1)
#             if w <= 0:
#                 continue
#             accum[idx] += mean_vec * w
#             weight_sum[idx, 0] += w

#         for idx, dist in info["right"]:
#             if idx in core_set:
#                 continue
#             w = 1.0 - dist / float(radius + 1)
#             if w <= 0:
#                 continue
#             accum[idx] += mean_vec * w
#             weight_sum[idx, 0] += w

#     mask = weight_sum[:, 0] > 0
#     out_disp[mask] += accum[mask] / weight_sum[mask]

#     return original_pts + out_disp


def apply_decayed_pull(original_pts, adjusted_pts, span_infos, radius=4, min_decay_mag=1.0):
    """
    Apply boundary-aware decayed pull.

    Left neighbors decay from min_decay_mag -> |disp(core[0])|
    Right neighbors decay from min_decay_mag -> |disp(core[-1])|

    Parameters
    ----------
    original_pts : (N, 2) ndarray
    adjusted_pts : (N, 2) ndarray
    span_infos : list[dict]
        Output of expand_neighborhood(...)
    radius : int
        Neighborhood radius
    min_decay_mag : float
        Minimum pull magnitude at the outermost decay point
    """
    original_pts = original_pts.astype(float)
    adjusted_pts = adjusted_pts.astype(float)

    disp = adjusted_pts - original_pts
    n = len(original_pts)

    out_disp = disp.copy()
    accum = np.zeros_like(disp)
    weight_sum = np.zeros((n, 1), dtype=float)

    core_set = set()
    for info in span_infos:
        core_set.update(info["core"])

    for info in span_infos:
        core = info["core"]
        if not core:
            continue

        left_vec = disp[core[0]]
        right_vec = disp[core[-1]]

        for idx, dist in info["left"]:
            if idx in core_set:
                continue

            if radius <= 1:
                t = 1.0
            else:
                t = 1.0 - (dist - 1) / float(radius - 1)   # dist=1 -> 1, dist=radius -> 0

            vec = t * left_vec

            accum[idx] += vec
            weight_sum[idx, 0] += 1.0

        for idx, dist in info["right"]:
            if idx in core_set:
                continue

            if radius <= 1:
                t = 1.0
            else:
                t = 1.0 - (dist - 1) / float(radius - 1)

            vec = t * right_vec

            accum[idx] += vec
            weight_sum[idx, 0] += 1.0

    mask = weight_sum[:, 0] > 0
    out_disp[mask] += accum[mask] / weight_sum[mask]

    return original_pts + out_disp

def smooth_core_displacement(original_pts, adjusted_pts, spans, passes=1):
    """
    Smooth displacement only inside each core span.
    Endpoints are kept unchanged for stability.
    """
    original_pts = original_pts.astype(float)
    adjusted_pts = adjusted_pts.astype(float)

    disp = adjusted_pts - original_pts
    out_disp = disp.copy()

    for span in spans:
        if len(span) < 3:
            continue

        span_disp = out_disp[span].copy()

        for _ in range(passes):
            new_span_disp = span_disp.copy()

            for i in range(1, len(span) - 1):
                new_span_disp[i] = (
                    0.33 * span_disp[i - 1]
                    + 0.33 * span_disp[i]
                    + 0.33 * span_disp[i + 1]
                )

            span_disp = new_span_disp

        out_disp[span] = span_disp

    return original_pts + out_disp

import numpy as np

def smooth_pull_magnitude_field(original_pts, adjusted_pts, spans, passes=2):
    """
    Smooth pull magnitude only, while keeping pull direction.

    Parameters
    ----------
    original_pts : (N, 2) ndarray
    adjusted_pts : (N, 2) ndarray
    spans : list[list[int]]
        Core spans.
    passes : int
        Number of smoothing passes.

    Returns
    -------
    new_pts : (N, 2) ndarray
    """
    original_pts = original_pts.astype(float)
    adjusted_pts = adjusted_pts.astype(float)

    disp = adjusted_pts - original_pts
    out_disp = disp.copy()

    for span in spans:
        if len(span) < 3:
            continue

        span_disp = out_disp[span].copy()
        mag = np.linalg.norm(span_disp, axis=1)

        # keep per-point direction
        direction = np.zeros_like(span_disp)
        nonzero = mag > 1e-8
        direction[nonzero] = span_disp[nonzero] / mag[nonzero][:, None]

        smooth_mag = mag.copy()

        for _ in range(passes):
            new_mag = smooth_mag.copy()
            for i in range(1, len(span) - 1):
                new_mag[i] = (
                    0.33 * smooth_mag[i - 1]
                    + 0.33 * smooth_mag[i]
                    + 0.33 * smooth_mag[i + 1]
                )
            smooth_mag = new_mag

        # reconstruct displacement from smoothed magnitude
        span_disp_new = direction * smooth_mag[:, None]
        out_disp[span] = span_disp_new

    return original_pts + out_disp

def refit_modified_spans(points, spans, anchor_size=3, passes=4, alpha=0.35):
    """
    Local refit on modified spans.

    Parameters
    ----------
    points : (N, 2) ndarray
        Current adjusted points.
    spans : list[list[int]]
        Core spans.
    anchor_size : int
        Extend each span by a few points on both sides as fixed anchors.
    passes : int
        Number of local smoothing/refit passes.
    alpha : float
        Refit strength.

    Returns
    -------
    new_pts : (N, 2) ndarray
    """
    pts = points.astype(float).copy()
    n = len(pts)

    for span in spans:
        if not span:
            continue

        start = (span[0] - anchor_size) % n
        end = (span[-1] + anchor_size) % n

        window = []
        i = start
        while True:
            window.append(i)
            if i == end:
                break
            i = (i + 1) % n
            if len(window) > n:
                break

        if len(window) < 3:
            continue

        local = pts[window].copy()

        for _ in range(passes):
            new_local = local.copy()

            # keep outer anchors fixed
            for j in range(1, len(local) - 1):
                avg_nb = 0.5 * (local[j - 1] + local[j + 1])
                new_local[j] = (1 - alpha) * local[j] + alpha * avg_nb

            local = new_local

        pts[window] = local

    return pts