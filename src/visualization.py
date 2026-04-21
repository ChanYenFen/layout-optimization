import matplotlib.pyplot as plt
import numpy as np

def _close_ring(points):
    """
    Append the first point to the end so matplotlib shows a closed loop.
    """
    if len(points) == 0:
        return points

    points = np.asarray(points, dtype=float)

    if np.allclose(points[0], points[-1]):
        return points

    return np.vstack([points, points[0]])


def visualize_adjusted_points(original_pts, adjusted_pts, moved_indices=None, title=None):
    """
    Debug view for a single contour.
    """
    original_closed = _close_ring(original_pts)
    adjusted_closed = _close_ring(adjusted_pts)

    plt.figure(figsize=(8, 8))
    plt.plot(
        original_closed[:, 0],
        original_closed[:, 1],
        "--",
        linewidth=1.0,
        label="original",
    )
    plt.plot(
        adjusted_closed[:, 0],
        adjusted_closed[:, 1],
        "-",
        linewidth=1.5,
        label="adjusted",
    )

    if moved_indices:
        moved = adjusted_pts[moved_indices]
        plt.scatter(moved[:, 0], moved[:, 1], s=8, label="moved points")

    if title:
        plt.title(title)

    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.legend()
    plt.show()

def visualize_multi_contours(
    binary_img,
    contour_results,
    title="All contours",
    draw_vectors=False,
):
    plt.figure(figsize=(8, 8), facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("white")

    for i, info in enumerate(contour_results):
        original_pts = info["points"]
        adjusted_pts = info["adjusted_points"]
        moved_indices = info.get("moved_indices", [])
        is_hole = info.get("is_hole", False)
        depth = info.get("depth", 0)

        original_closed = _close_ring(original_pts)
        adjusted_closed = _close_ring(adjusted_pts)

        plt.plot(
            original_closed[:, 0],
            original_closed[:, 1],
            "--",
            linewidth=0.8,
        )
        plt.plot(
            adjusted_closed[:, 0],
            adjusted_closed[:, 1],
            "-",
            linewidth=1.5,
        )

        if moved_indices:
            moved = adjusted_pts[moved_indices]
            plt.scatter(moved[:, 0], moved[:, 1], s=6)

        if draw_vectors:
            for item in info.get("debug_vectors", []):
                p = item["point"]
                disp = item["disp"]

                plt.arrow(
                    p[0],
                    p[1],
                    disp[0],
                    disp[1],
                    head_width=2.0,
                    length_includes_head=True,
                )

        center = np.mean(original_pts, axis=0)
        plt.text(
            center[0],
            center[1],
            f"{i} | hole={is_hole} | d={depth}",
            fontsize=8,
        )

    plt.title(title)
    plt.axis("equal")
    plt.show()