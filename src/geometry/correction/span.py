
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