import numpy as np


def _seg_intersect(p1: np.ndarray, p2: np.ndarray,
                   p3: np.ndarray, p4: np.ndarray):
    """Return the intersection point of segments p1-p2 and p3-p4, or None."""
    d1 = p2 - p1
    d2 = p4 - p3
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-10:
        return None
    diff = p3 - p1
    t = (diff[0] * d2[1] - diff[1] * d2[0]) / cross
    u = (diff[0] * d1[1] - diff[1] * d1[0]) / cross
    if 0.0 < t < 1.0 and 0.0 < u < 1.0:
        return p1 + t * d1
    return None


def resolve_self_intersections(pts: np.ndarray, max_passes: int = 30) -> np.ndarray:
    """
    Remove self-intersecting loops from a closed contour.

    For each pair of non-adjacent crossing edges (i→i+1) and (j→j+1),
    the shorter arc between them is a spurious loop.  Replace both edges
    with the intersection point and drop the enclosed vertices.
    Repeats until no crossings remain or max_passes is reached.

    Parameters
    ----------
    pts : (N, 2) float64 ndarray
    max_passes : int
        Safety cap on iterations.

    Returns
    -------
    (M, 2) ndarray  (M <= N)
    """
    pts = pts.copy()

    for _ in range(max_passes):
        n = len(pts)
        if n < 4:
            break

        fixed = False
        for i in range(n):
            a1 = pts[i]
            a2 = pts[(i + 1) % n]

            # j must be non-adjacent to i; also skip the wrap-around pair (0, n-1)
            j_end = n - (1 if i == 0 else 0)
            for j in range(i + 2, j_end):
                b1 = pts[j]
                b2 = pts[(j + 1) % n]

                ip = _seg_intersect(a1, a2, b1, b2)
                if ip is None:
                    continue

                # Inner arc is pts[i+1 .. j].  Outer arc is everything else.
                # Always remove the inner arc (it is the spurious loop).
                inner_len = j - i - 1
                outer_len = n - j - 1 + i  # indices j+1..n-1 + 0..i-1

                if inner_len <= outer_len:
                    # Remove indices i+1 .. j, insert intersection point after i
                    pts = np.vstack([pts[: i + 1], ip, pts[j + 1 :]])
                else:
                    # The outer arc is shorter — remove it instead.
                    # Keep indices i+1 .. j, prepend/append intersection point.
                    pts = np.vstack([ip, pts[i + 1 : j + 1]])

                fixed = True
                break

            if fixed:
                break

        if not fixed:
            break

    return pts


def fillet_sharp_corners(
    pts: np.ndarray,
    min_turn_deg: float = 90.0,
    radius: float = 5.0,
    n_arc: int = 4,
) -> np.ndarray:
    """
    Smooth sharp corners with a quadratic Bézier fillet.

    For each vertex where the turning angle exceeds `min_turn_deg`, the
    corner vertex is replaced by:
      - P1 : a point `radius` pixels back along the incoming edge
      - arc : `n_arc` points on the quadratic Bézier (P1, V as control, P2)
      - P2 : a point `radius` pixels forward along the outgoing edge

    The original corner vertex V is removed.

    Parameters
    ----------
    pts          : (N, 2) float64 closed contour
    min_turn_deg : fillet corners whose turn angle exceeds this value (degrees)
    radius       : fillet radius in the same units as pts (pixels)
    n_arc        : number of interior arc points inserted per corner

    Returns
    -------
    (M, 2) ndarray  (M >= N when corners are filleted)
    """
    import math

    cos_threshold = math.cos(math.radians(min_turn_deg))
    n = len(pts)
    new_pts: list[np.ndarray] = []

    for i in range(n):
        V = pts[i].astype(float)
        A = pts[(i - 1) % n].astype(float)
        B = pts[(i + 1) % n].astype(float)

        d_in = V - A   # incoming edge direction
        d_out = B - V  # outgoing edge direction
        len_in = np.linalg.norm(d_in)
        len_out = np.linalg.norm(d_out)

        if len_in < 1e-8 or len_out < 1e-8:
            new_pts.append(V)
            continue

        cos_turn = np.dot(d_in / len_in, d_out / len_out)

        if cos_turn >= cos_threshold:
            new_pts.append(V)
            continue

        # Clamp radius so the fillet points stay on their respective edges.
        r = min(radius, len_in * 0.45, len_out * 0.45)
        P1 = V - (d_in / len_in) * r   # step back from V along incoming edge
        P2 = V + (d_out / len_out) * r  # step forward from V along outgoing edge

        # Quadratic Bézier: P1 → (V as control point) → P2
        arc = [
            (1 - t) ** 2 * P1 + 2 * t * (1 - t) * V + t ** 2 * P2
            for t in np.linspace(0.0, 1.0, n_arc + 2)[1:-1]
        ]

        new_pts.append(P1)
        new_pts.extend(arc)
        new_pts.append(P2)

    return np.array(new_pts, dtype=float)
