import matplotlib.pyplot as plt


def visualize_points(points, title="Points"):
    x = points[:, 0]
    y = points[:, 1]

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=1)

    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.title(title)

    plt.show()


def visualize_adjusted_points(points, adjusted_points, indices, title="Adjusted Points"):
    plt.figure(figsize=(6, 6))

    plt.scatter(points[:, 0], points[:, 1], s=1, label="original")
    plt.scatter(adjusted_points[:, 0], adjusted_points[:, 1], s=1, label="adjusted")

    if indices:
        idx = list(indices)
        plt.scatter(
            adjusted_points[idx, 0],
            adjusted_points[idx, 1],
            s=8,
            label="modified"
        )

    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.title(title)
    plt.legend()
    plt.show()