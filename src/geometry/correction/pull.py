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
# Pull function
# -----------------------------------------------------------------------------

def pull_points(
    points: np.ndarray,
    binary_img: np.ndarray,
    min_distance: float = 5,
    increment: float = 5,
    verbose: bool = False,
) -> tuple[np.ndarray, list[int], list[dict]]:
    """
    Pull boundary points outward when the opposite side of the shape
    is found too close along the inward normal direction.

    Parameters
    ----------
    points : (N, 2) ndarray
        Input contour points.
    binary_img : (H, W) ndarray
        Binary image with values 0 / 255.
    min_distance : float
        Minimum allowed local thickness.
    increment : float
        Maximum pull distance for one point.
    verbose : bool
        If True, print debug information.

    Returns
    -------
    adjusted : (N, 2) ndarray
        Adjusted contour points.
    indices_to_adjust : list[int]
        Indices of points that were directly pulled.
    debug_vectors : list[dict]
        Per-point pull debug info containing:
        - index
        - point
        - inward
        - outward
        - disp
        - pull_dist
    """
    adjusted = points.astype(np.float64).copy()
    indices_to_adjust: list[int] = []
    debug_vectors: list[dict] = []

    h, w = binary_img.shape
    max_steps = int(np.ceil(min_distance))

    for i in range(len(points)):
        inward_normal = get_inward_normal(points, i, binary_img)

        if verbose:
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

            # still in background / void: keep walking inward
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

                    if verbose:
                        print(f"[{i}] distance = {distance}")

                    if distance < min_distance:
                        violation = max(min_distance - distance, 0.0)
                        pull_dist = min(np.sqrt(violation) * 1.2, increment)
                        # Alternative mapping:
                        # pull_dist = min((min_distance - distance) / 2.0, increment)

                        adjusted[i] = p + outward_normal * pull_dist
                        indices_to_adjust.append(i)

                        debug_vectors.append(
                            {
                                "index": i,
                                "point": p.copy(),
                                "inward": inward_normal.copy(),
                                "outward": outward_normal.copy(),
                                "disp": (adjusted[i] - p).copy(),
                                "pull_dist": float(pull_dist),
                            }
                        )

                        if verbose:
                            print(
                                f"PULL [{i}] distance={distance}, "
                                f"pull_dist={pull_dist}, "
                                f"from={p} to={adjusted[i]}"
                            )

                    opposite_found = True
                    break

            if opposite_found:
                break

    return adjusted, indices_to_adjust, debug_vectors
