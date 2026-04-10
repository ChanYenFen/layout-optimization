import numpy as np


def compute_tangent(points: np.ndarray, i: int) -> np.ndarray:
    """
    Compute the local tangent direction at point i using its previous and next neighbors.
    """
    prev_idx = (i - 1) % len(points)
    next_idx = (i + 1) % len(points)

    tangent = points[next_idx] - points[prev_idx]
    norm = np.linalg.norm(tangent)

    if norm == 0:
        return np.array([0.0, 0.0], dtype=np.float64)

    return tangent / norm


def compute_normal(tangent: np.ndarray) -> np.ndarray:
    """
    Compute a perpendicular normal from a 2D tangent vector.
    """
    return np.array([-tangent[1], tangent[0]], dtype=np.float64)


def get_inward_normal(points: np.ndarray, i: int, binary_img: np.ndarray) -> np.ndarray:
    """
    Determine the inward normal direction by testing both candidate normals
    against the binary image.

    White (255) is assumed to be the valid interior region.
    """
    h, w = binary_img.shape

    tangent = compute_tangent(points, i)
    if np.linalg.norm(tangent) == 0:
        return np.array([0.0, 0.0], dtype=np.float64)

    n1 = compute_normal(tangent)
    n2 = -n1

    p = points[i].astype(np.float64)

    def sample(pt: np.ndarray) -> int:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < w and 0 <= y < h:
            return int(binary_img[y, x])
        return 0

    if sample(p + n1) > 0:
        return n1
    if sample(p + n2) > 0:
        return n2

    return np.array([0.0, 0.0], dtype=np.float64)


def pull_points(
    points: np.ndarray,
    binary_img: np.ndarray,
    min_distance: float = 5,
    increment: float = 5,
) -> tuple[np.ndarray, list[int]]:
    """
    Pull boundary points outward when the opposite side of the shape is found
    too close along the inward normal direction.

    Parameters
    ----------
    points : np.ndarray
        Contour points with shape (N, 2).
    binary_img : np.ndarray
        Binary image where white (255) indicates the valid region.
    min_distance : float
        Minimum allowed distance across the local shape thickness.
    increment : float
        Maximum adjustment distance applied in one iteration.

    Returns
    -------
    adjusted : np.ndarray
        Updated point array after correction.
    indices_to_adjust : list[int]
        Indices of points that were adjusted.
    """
    adjusted = points.astype(np.float64).copy()
    indices_to_adjust: list[int] = []

    h, w = binary_img.shape
    max_steps = int(np.ceil(min_distance))

    for i in range(len(points)):
        inward_normal = get_inward_normal(points, i, binary_img)
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
                    if distance < min_distance:
                        pull_dist = min((min_distance - distance) / 2.0, increment)
                        adjusted[i] = p + outward_normal * pull_dist
                        indices_to_adjust.append(i)
                    opposite_found = True
                    break

            if opposite_found:
                break

    return adjusted, indices_to_adjust