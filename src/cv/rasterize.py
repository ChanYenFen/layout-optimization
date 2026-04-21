import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Image loading
# -----------------------------------------------------------------------------

def load_binary_image(path, threshold=127, invert=False):
    """
    Load an image as grayscale and convert it to a binary image (0 / 255).

    Parameters
    ----------
    path : str
        Image path.
    threshold : int
        Threshold value for binarization.
    invert : bool
        If True, use THRESH_BINARY_INV.

    Returns
    -------
    binary : (H, W) ndarray
        Binary image with values 0 and 255.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")

    thresh_mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(img, threshold, 255, thresh_mode)
    return binary


# -----------------------------------------------------------------------------
# Contour extraction helpers
# -----------------------------------------------------------------------------

def _contour_depth(hierarchy, idx):
    """
    Compute nesting depth from OpenCV contour hierarchy.

    depth = 0  -> outer contour
    depth = 1  -> hole inside outer contour
    depth = 2  -> island inside hole
    ...
    """
    depth = 0
    parent = hierarchy[idx][3]

    while parent != -1:
        depth += 1
        parent = hierarchy[parent][3]

    return depth


# -----------------------------------------------------------------------------
# Multi-contour API (preferred)
# -----------------------------------------------------------------------------

def extract_contours(binary_img, min_points=8, min_area=4.0, debug=False):
    """
    Extract all contour rings from a binary image, including inner hole boundaries.

    Parameters
    ----------
    binary_img : (H, W) ndarray
        Binary image with values 0 / 255.
    min_points : int
        Minimum number of contour points to keep.
    min_area : float
        Minimum absolute contour area to keep.
    debug : bool
        If True, print raw and kept contour information.

    Returns
    -------
    contour_infos : list[dict]
        Each item contains:
        - id: contour index from OpenCV
        - points: (N, 2) ndarray
        - depth: nesting depth in hierarchy
        - is_hole: bool
        - parent: parent contour index
        - area: absolute contour area
    """
    contours, hierarchy = cv2.findContours(
        binary_img,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        raise ValueError("No contours found in the image.")

    if hierarchy is None:
        raise ValueError("Contour hierarchy is missing.")

    hierarchy = hierarchy[0]

    if debug:
        print(f"raw contours count = {len(contours)}")
        for i, contour in enumerate(contours):
            pts_n = len(contour.reshape(-1, 2))
            area = abs(cv2.contourArea(contour))
            nxt, prev, child, parent = hierarchy[i]
            print(
                f"[raw {i}] points={pts_n}, area={area:.1f}, "
                f"next={nxt}, prev={prev}, child={child}, parent={parent}"
            )

    contour_infos = []

    for i, contour in enumerate(contours):
        points = contour.reshape(-1, 2)

        if len(points) < min_points:
            continue

        area = abs(cv2.contourArea(contour))
        if area < min_area:
            continue

        depth = _contour_depth(hierarchy, i)

        info = {
            "id": i,
            "points": points,
            "depth": depth,
            "is_hole": (depth % 2 == 1),
            "parent": int(hierarchy[i][3]),
            "area": float(area),
        }
        contour_infos.append(info)

        if debug:
            print(
                f"[keep {i}] points={len(points)}, area={area:.1f}, "
                f"depth={depth}, is_hole={info['is_hole']}"
            )

    contour_infos.sort(key=lambda x: (x["depth"], -x["area"]))
    return contour_infos


# -----------------------------------------------------------------------------
# Backward-compatible single-contour API (temporary)
# -----------------------------------------------------------------------------

def extract_contour(binary_img, min_points=8, min_area=4.0, debug=False):
    """
    Backward-compatible wrapper.

    Returns the first non-hole contour from extract_contours(...).
    Keep this temporarily while refactoring main.py / older scripts.
    """
    contour_infos = extract_contours(
        binary_img,
        min_points=min_points,
        min_area=min_area,
        debug=debug,
    )

    for info in contour_infos:
        if not info["is_hole"]:
            return info["points"]

    raise ValueError("No outer contour found.")