import matplotlib.pyplot as plt
import numpy as np


def visualize_points(points, title="Points"):
    x = points[:, 0]
    y = points[:, 1]

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=2)

    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.title(title)
    plt.show()


def visualize_adjusted_points(points, adjusted_points, indices, title="Adjusted Points"):
    plt.figure(figsize=(6, 6))

    # original
    plt.scatter(points[:, 0], points[:, 1], s=2, alpha=0.5, label="original")

    # adjusted
    plt.scatter(adjusted_points[:, 0], adjusted_points[:, 1], s=2, alpha=0.7, label="adjusted")

    # modified only
    if indices:
        idx = np.array(list(indices), dtype=int)

        plt.scatter(
            adjusted_points[idx, 0],
            adjusted_points[idx, 1],
            s=16,
            label="modified"
        )

        # optional: draw displacement lines
        for i in idx:
            plt.plot(
                [points[i, 0], adjusted_points[i, 0]],
                [points[i, 1], adjusted_points[i, 1]],
                linewidth=0.5,
                alpha=0.6
            )

    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.title(title)
    plt.legend()
    plt.show()


def visualize_pipeline(original, adjusted, smoothed, indices=None, title="Pipeline Comparison"):
    plt.figure(figsize=(7, 7))

    plt.scatter(original[:, 0], original[:, 1], s=2, alpha=0.4, label="original")
    plt.scatter(adjusted[:, 0], adjusted[:, 1], s=2, alpha=0.6, label="adjusted")
    plt.scatter(smoothed[:, 0], smoothed[:, 1], s=2, alpha=0.8, label="smoothed")

    if indices:
        idx = np.array(list(indices), dtype=int)
        plt.scatter(
            adjusted[idx, 0],
            adjusted[idx, 1],
            s=18,
            label="modified"
        )

    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.title(title)
    plt.legend()
    plt.show()

