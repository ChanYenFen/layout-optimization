import numpy as np
import matplotlib.pyplot as plt

from cv.rasterize import load_binary_image, extract_contours
from geometry.correction.pull import get_inward_normal

print("debug_width.py started")

IMAGE_PATH = "examples/simple_case_1.png"

CONTOUR_INDEX = 0

MIN_DISTANCE = 50
SEARCH_DEPTH = 50
INCREMENT = 5

NORMAL_STRIDE = 10
NORMAL_SCALE = 10


def measure_local_widths(points, binary_img):
    """
    Measure local width along inward normals.
    This is debug-only. It does not modify points.
    """
    h, w = binary_img.shape[:2]
    records = []

    for i, p in enumerate(points):
        p = p.astype(float)

        normal = get_inward_normal(points, i, binary_img)

        if np.linalg.norm(normal) < 1e-8:
            records.append({
                "idx": i,
                "point": p,
                "normal": np.zeros(2),
                "normal_valid": False,
                "hit": False,
                "hit_point": None,
                "distance": np.nan,
                "violation": 0.0,
                "will_pull": False,
                "pull_dist": 0.0,
                "reason": "no_normal",
            })
            continue

        normal = normal / np.linalg.norm(normal)

        hit = False
        hit_point = None
        distance = np.nan
        reason = "no_hit"

        for d in range(1, SEARCH_DEPTH + 1):
            probe = p + normal * d
            x, y = int(round(probe[0])), int(round(probe[1]))

            if x < 0 or x >= w or y < 0 or y >= h:
                reason = "out_of_bounds"
                break

            if binary_img[y, x] == 0:
                hit = True
                hit_point = probe
                distance = float(d)
                reason = "hit"
                break

        if hit:
            violation = max(MIN_DISTANCE - distance, 0.0)
            will_pull = distance < MIN_DISTANCE
            pull_dist = min(violation / 2.0, INCREMENT)
        else:
            violation = 0.0
            will_pull = False
            pull_dist = 0.0

        records.append({
            "idx": i,
            "point": p,
            "normal": normal,
            "normal_valid": True,
            "hit": hit,
            "hit_point": hit_point,
            "distance": distance,
            "violation": violation,
            "will_pull": will_pull,
            "pull_dist": pull_dist,
            "reason": reason,
        })

    return records

def extract_points_from_contour_data(contours, contour_index=0):
    """
    Extract point array from selected contour info.
    Debug-only helper.
    """
    print("type(contours):", type(contours))
    print("len(contours):", len(contours))

    print("available contours:")
    for k, c in enumerate(contours):
        if isinstance(c, dict):
            print(
                f"[{k}] id={c.get('id')}, "
                f"points={len(c.get('points'))}, "
                f"depth={c.get('depth')}, "
                f"is_hole={c.get('is_hole')}, "
                f"area={c.get('area'):.1f}"
            )

    first = contours[contour_index]
    print("selected contour index:", contour_index)
    print("type(selected):", type(first))

    if isinstance(first, dict):
        print("selected keys:", first.keys())

        pts = first["points"]
        pts = np.asarray(pts).squeeze()

        print("selected id:", first["id"])
        print("selected is_hole:", first["is_hole"])
        print("selected depth:", first["depth"])
        print("pts shape:", pts.shape)

        return pts.astype(float)

    pts = np.asarray(first).squeeze()
    print("pts shape:", pts.shape)
    return pts[:, :2].astype(float)

def print_summary(records):
    total = len(records)
    hit = sum(r["hit"] for r in records)
    will_pull = sum(r["will_pull"] for r in records)
    invalid = sum(not r["normal_valid"] for r in records)
    no_hit = sum(r["reason"] == "no_hit" for r in records)
    out_of_bounds = sum(r["reason"] == "out_of_bounds" for r in records)

    distances = [
        r["distance"]
        for r in records
        if r["hit"] and np.isfinite(r["distance"])
    ]

    print("=== width debug summary ===")
    print(f"total points     : {total}")
    print(f"hit count        : {hit}")
    print(f"will pull count  : {will_pull}")
    print(f"invalid normals  : {invalid}")
    print(f"no hit           : {no_hit}")
    print(f"out of bounds    : {out_of_bounds}")

    if distances:
        print(f"min distance     : {np.min(distances):.2f}")
        print(f"max distance     : {np.max(distances):.2f}")
        print(f"mean distance    : {np.mean(distances):.2f}")


def visualize_width_debug(binary_img, points, records):
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(binary_img, cmap="gray", origin="upper")

    pts = np.asarray(points)

    # contour
    ax.plot(
        pts[:, 0],
        pts[:, 1],
        linewidth=1.0,
        color="lightgray",
        label="contour",
        zorder=1,
    )

    hit_pts = []
    hit_distances = []
    pull_pts = []
    failed_pts = []

    for r in records:
        p = r["point"]

        if r["hit"] and np.isfinite(r["distance"]):
            hit_pts.append(p)
            hit_distances.append(r["distance"])

            if r["will_pull"]:
                pull_pts.append(p)
        else:
            failed_pts.append(p)

    # distance heatmap
    if hit_pts:
        hit_pts = np.asarray(hit_pts)
        hit_distances = np.asarray(hit_distances)

        sc = ax.scatter(
            hit_pts[:, 0],
            hit_pts[:, 1],
            c=hit_distances,
            s=12,
            cmap="viridis",
            label="measured width",
            zorder=3,
        )
        fig.colorbar(sc, ax=ax, label="distance")

    # candidate pulled points
    if pull_pts:
        pull_pts = np.asarray(pull_pts)
        ax.scatter(
            pull_pts[:, 0],
            pull_pts[:, 1],
            s=36,
            color="red",
            label="will pull",
            zorder=5,
        )

    # failed points
    if failed_pts:
        failed_pts = np.asarray(failed_pts)
        ax.scatter(
            failed_pts[:, 0],
            failed_pts[:, 1],
            s=24,
            color="black",
            marker="x",
            label="no hit / invalid",
            zorder=6,
        )

    # sampled normals
    for r in records[::NORMAL_STRIDE]:
        if not r["normal_valid"]:
            continue

        p = r["point"]
        n = r["normal"]

        ax.arrow(
            p[0],
            p[1],
            n[0] * NORMAL_SCALE,
            n[1] * NORMAL_SCALE,
            head_width=2.0,
            head_length=3.0,
            color="black",
            linewidth=0.5,
            alpha=0.8,
            zorder=4,
        )

    ax.set_title("Width Detector Debug")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def main():
    img = load_binary_image(IMAGE_PATH)
    contours = extract_contours(img)

    if not contours:
        print("No contour found.")
        return

    pts = extract_points_from_contour_data(contours, CONTOUR_INDEX)

    records = measure_local_widths(pts, img)

    print_summary(records)
    visualize_width_debug(img, pts, records)


if __name__ == "__main__":
    main()