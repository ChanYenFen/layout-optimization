import numpy as np

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