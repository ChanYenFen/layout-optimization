import cv2
import numpy as np

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
  """""
