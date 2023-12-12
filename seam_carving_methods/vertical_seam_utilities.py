import cv2
import numpy as np
from numba import jit
### vertical seam carving utility methods

@jit
def compute_backward_energy_vertical(image):
    # Compute the backward energy matrix using Sobel operators
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gradient_x = cv2.Sobel(src=gray_image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    gradient_y = cv2.Sobel(src=gray_image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    energy_map = np.abs(gradient_x) + np.abs(gradient_y)

    cumulative_energy = np.copy(energy_map)
    height, width = cumulative_energy.shape

    # Calculate the cumulative minimum energy
    for h in range(1, height):
        cumulative_energy[h, 1:width - 1] += np.min(np.array([np.roll(cumulative_energy[h - 1], 1),
                                                               cumulative_energy[h - 1], np.roll(cumulative_energy[h - 1], -1)]), axis=0)[1:width - 1]
        cumulative_energy[h, 0] += min(cumulative_energy[h - 1, 0], cumulative_energy[h - 1, 1])
        cumulative_energy[h, width - 1] += min(cumulative_energy[h - 1, width - 2], cumulative_energy[h - 1, width - 1])

    return cumulative_energy

@jit
def compute_forward_energy_vertical(image):
    # Compute the forward energy matrix
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    cumulative_energy = np.zeros(gray_image.shape)

    neighbor_left = np.roll(gray_image, 1, axis=1)
    neighbor_up = np.roll(gray_image, 1, axis=0)
    neighbor_right = np.roll(gray_image, -1, axis=1)

    energy_left_right = np.abs(neighbor_right - neighbor_left)
    energy_up_left_right = energy_left_right + np.abs(neighbor_up - neighbor_left)
    energy_up_right = energy_left_right + np.abs(neighbor_up - neighbor_right)

    for i in range(1, gray_image.shape[0]):
        cumulative_energy[i] = np.min(np.array([np.roll(cumulative_energy[i - 1], 1) + energy_up_left_right[i],
                                                cumulative_energy[i - 1] + energy_left_right[i],
                                                np.roll(cumulative_energy[i - 1], -1) + energy_up_right[i]]), axis=0)

    return cumulative_energy

@jit
def find_vertical_seam(cost_matrix):
    # Find the vertical seam indices using dynamic programming
    height, width = cost_matrix.shape
    seam_indices = [np.argmin(cost_matrix[height - 1])]

    for h in range(height - 2, -1, -1):
        current_index = seam_indices[-1]
        if current_index == 0:
            seam_indices.append(np.argmin(cost_matrix[h, 0:2]))
        elif current_index == width - 1:
            seam_indices.append(width - 2 + np.argmin(cost_matrix[h, width - 2:width]))
        else:
            seam_indices.append(current_index - 1 + np.argmin(cost_matrix[h, current_index - 1:current_index + 2]))

    seam_indices.reverse()
    return seam_indices

@jit
def remove_vertical_seam(image, seam_indices):
    # Remove the seam from the image
    if len(image.shape) == 3:
        output = np.zeros([image.shape[0], image.shape[1] - 1, image.shape[2]], dtype=image.dtype)
    else:
        output = np.zeros([image.shape[0], image.shape[1] - 1], dtype=image.dtype)

    for h in range(image.shape[0]):
        current_index = seam_indices[h]
        output[h, :current_index] = image[h, :current_index]
        output[h, current_index:] = image[h, current_index + 1:]

    return output

