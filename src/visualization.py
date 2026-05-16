import matplotlib.pyplot as plt
import numpy as np


def _close_ring(points):
    if len(points) == 0:
        return points
    points = np.asarray(points, dtype=float)
    if np.allclose(points[0], points[-1]):
        return points
    return np.vstack([points, points[0]])


def visualize_adjusted_points(original_pts, adjusted_pts, moved_indices=None, title=None):
    """Debug view for a single contour."""
    original_closed = _close_ring(original_pts)
    adjusted_closed = _close_ring(adjusted_pts)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    ax.plot(
        original_closed[:, 0],
        original_closed[:, 1],
        "--",
        linewidth=1.0,
        color="yellow",
        label="original",
    )
    ax.plot(
        adjusted_closed[:, 0],
        adjusted_closed[:, 1],
        "-",
        linewidth=1.5,
        color="white",
        label="adjusted",
    )

    n = min(len(original_pts), len(adjusted_pts))
    disp = np.linalg.norm(adjusted_pts[:n] - original_pts[:n], axis=1)
    moved_pts = adjusted_pts[:n][disp > 1e-8]
    if len(moved_pts):
        ax.scatter(moved_pts[:, 0], moved_pts[:, 1], s=8, color="orangered", label="moved points")

    if title:
        ax.set_title(title, color="white")

    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("white")

    legend = ax.legend()
    legend.get_frame().set_facecolor("black")
    legend.get_frame().set_edgecolor("white")
    for text in legend.get_texts():
        text.set_color("white")

    plt.show()


def visualize_multi_contours(
    binary_img,
    contour_results,
    title="All contours",
    draw_vectors=False,
):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    for i, info in enumerate(contour_results):
        original_pts = info["points"]
        adjusted_pts = info["adjusted_points"]
        moved_indices = info.get("moved_indices", [])
        is_hole = info.get("is_hole", False)
        depth = info.get("depth", 0)

        original_closed = _close_ring(original_pts)
        adjusted_closed = _close_ring(adjusted_pts)

        ax.plot(
            original_closed[:, 0],
            original_closed[:, 1],
            "--",
            linewidth=0.8,
            color="yellow",
        )
        ax.plot(
            adjusted_closed[:, 0],
            adjusted_closed[:, 1],
            "-",
            linewidth=1.5,
            color="white",
        )

        n = min(len(original_pts), len(adjusted_pts))
        disp = np.linalg.norm(adjusted_pts[:n] - original_pts[:n], axis=1)
        moved_pts = adjusted_pts[:n][disp > 1e-8]
        if len(moved_pts):
            ax.scatter(moved_pts[:, 0], moved_pts[:, 1], s=6, color="orangered")

        if draw_vectors:
            for item in info.get("debug_vectors", []):
                p = item["point"]
                disp = item["disp"]
                ax.arrow(
                    p[0],
                    p[1],
                    disp[0],
                    disp[1],
                    head_width=0.4,
                    head_length=0.4,
                    linewidth=0.6,
                    color="white",
                    length_includes_head=True,
                )

        # Place label just outside the bottom-right corner of the bounding box.
        # y-axis is inverted so y_max is the visual bottom; adding 6px pushes below it.
        x_label = original_pts[:, 0].max() + 6
        y_label = original_pts[:, 1].max() + 6
        ax.text(
            x_label,
            y_label,
            f"{i} | hole={is_hole} | d={depth}",
            fontsize=6,
            ha="left",
            va="top",
            color="white",
        )

    ax.set_title(title, color="white")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("white")

    plt.show()
