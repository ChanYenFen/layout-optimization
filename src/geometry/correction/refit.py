
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