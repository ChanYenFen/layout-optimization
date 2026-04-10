from cv.rasterize import load_binary_image, extract_contour
from geometry.correction import pull_points
from visualization import visualize_points, visualize_adjusted_points
import geometry.correction as correction

def main():
    img = load_binary_image("examples/simple_case.png")
    pts = extract_contour(img)

    adjusted_pts, indices = pull_points(
        pts,
        img,
        min_distance=15,
        increment=10
    )

    # visualize_points(pts, title="Original Contour")
    # visualize_points(adjusted_pts, title="Adjusted Contour")
    visualize_adjusted_points(pts, adjusted_pts, indices, title="Pull Points Debug")

    print(f"Loaded {len(pts)} contour points")
    print(f"Adjusted {len(indices)} points")


if __name__ == "__main__":
    main()