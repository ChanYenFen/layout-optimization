import cv2
import numpy as np
import numpy as np

print(cv2.__version__)
print(np.__version__)
def load_binary_image(path, threshold=127):
  """
  load image and convert to binary (0, 255)
  """
  img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  if img is None:
    raise ValueError(f"Failed to load image: {path}")
  _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

  return binary

def extract_contour(binary_img):
  """"
  extract contour points from binary image
   - input: binary image (0, 255)
   - output: list of contour points (x, y) 
  """
  contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  
  if not contours:
    raise ValueError("No contours found in the image.")
  
  contour = contours[0]  # Assuming we want the largest contour
  points = contour.reshape(-1, 2)

  return points
