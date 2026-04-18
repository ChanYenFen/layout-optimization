import numpy as np

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