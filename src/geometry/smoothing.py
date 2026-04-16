import numpy as np

def laplacian_smoothing(points, fixed_indices=None, iterations=10, alpha=0.2):
    pts = points.astype(float).copy()
    n = len(pts)

    if fixed_indices is None:
        fixed_indices = set()
    else:
        fixed_indices = set(fixed_indices)

    for _ in range(iterations):
        new_pts = pts.copy()

        for i in range(n):
            if i in fixed_indices:
                continue

            prev_pt = pts[(i - 1) % n]
            next_pt = pts[(i + 1) % n]
            avg_neighbor = 0.5 * (prev_pt + next_pt)

            new_pts[i] = pts[i] + alpha * (avg_neighbor - pts[i])

        pts = new_pts

    return pts