import numpy as np

import hough_transformations as ht
from canny_edge_detection import detect_edges

# grayscale coefficients
GRAYSCALE_COEFFICIENTS = [0.2989, 0.5870, 0.1140]


def rgb_to_grayscale(image: np.array) -> np.array:
    return np.dot(image[..., :3], GRAYSCALE_COEFFICIENTS)


def detect_lines(img: np.array, num_of_lines: np.int32 = 30):
    # convert rgb to grayscale
    gray_img = rgb_to_grayscale(img)

    # apply canny edge detector
    edges = detect_edges(gray_img, 5, 20)
    # apply hough transform to detect lines
    acc, thetas, rhos = ht.hough_line(edges)

    lines_indices = ht.get_top_n_lines_indices_in_matrix(acc, num_of_lines)

    for row_idx, col_idx in lines_indices:
        img = ht.create_hough_line_in_img(img, rhos, thetas, row_idx, col_idx)

    return img
