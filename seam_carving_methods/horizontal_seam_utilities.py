import cv2
import numpy as np
from numba import jit

@jit
def compute_forward_energy_horizontal(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    height, width = gray_image.shape
    cumulative_energy = np.zeros((height, width))

    # Calculate the differences between neighboring pixels
    neighbor_left = np.roll(gray_image, 1, axis=1)
    neighbor_up = np.roll(gray_image, 1, axis=0)
    neighbor_right = np.roll(gray_image, -1, axis=1)

    # Calculate the energy for each direction
    energy_left_right = np.abs(neighbor_right - neighbor_left)
    energy_up_left = np.abs(neighbor_up - neighbor_left)
    energy_up_right = np.abs(neighbor_up - neighbor_right)

    # Compute the forward energy for each pixel
    for w in range(1, width):
        for h in range(height):
            up_energy = energy_left_right[h, w]
            if h != 0:
                up_energy += energy_up_left[h, w]
            if h != height - 1:
                up_energy += energy_up_right[h, w]

            if h == 0:
                cumulative_energy[h, w] = up_energy + cumulative_energy[h, w - 1]
            elif h == height - 1:
                cumulative_energy[h, w] = up_energy + min(cumulative_energy[h, w - 1], cumulative_energy[h - 1, w - 1])
            else:
                cumulative_energy[h, w] = up_energy + min(cumulative_energy[h - 1, w - 1], cumulative_energy[h, w - 1], cumulative_energy[h + 1, w - 1])

    return cumulative_energy

@jit
def compute_backward_energy_horizontal(image):
    """ Compute the backward energy for horizontal seam carving using Sobel operators. """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    energy_map = np.abs(grad_x) + np.abs(grad_y)

    cumulative_energy = np.copy(energy_map)
    rows, cols = energy_map.shape

    for col in range(1, cols):
        for row in range(rows):
            upper = max(row - 1, 0)
            lower = min(row + 1, rows - 1)
            neighbors = [
                cumulative_energy[upper, col - 1],
                cumulative_energy[row, col - 1],
                cumulative_energy[lower, col - 1]
            ]
            cumulative_energy[row, col] += min(neighbors)

    return cumulative_energy

@jit
def find_horizontal_seam(cost_matrix):
    """ Find the horizontal seam in an image. """
    rows, cols = cost_matrix.shape
    seam = [np.argmin(cost_matrix[:, cols - 1])]

    for col in reversed(range(cols - 1)):
        row = seam[-1]
        start, end = max(0, row - 1), min(rows, row + 2)
        seam.append(start + np.argmin(cost_matrix[start:end, col]))

    return list(reversed(seam))

def remove_horizontal_seam(image, seam_indices):
    """ Remove a horizontal seam from an image. """
    rows, cols = image.shape[:2]
    output = np.zeros((rows - 1, cols, image.shape[2]), dtype=image.dtype) if image.ndim == 3 else np.zeros((rows - 1, cols), dtype=image.dtype)

    for col in range(cols):
        row = seam_indices[col]
        output[:, col] = np.concatenate([image[:row, col], image[row + 1:, col]])

    return output

