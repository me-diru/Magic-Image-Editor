
#adaptive removal
import random
import time
import numpy as np
import cv2
from numba import jit
##### Vertical Functions
@jit
def forward_energy_matrix(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gradient_x = cv2.Sobel(src=gray_image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    gradient_y = cv2.Sobel(src=gray_image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    energy_map = np.abs(gradient_x) + np.abs(gradient_y)

    height, width = energy_map.shape
    cumulative_energy = np.copy(energy_map)

    for h in range(height - 1, -1, -1):  # Start from the last row and move upward
        for w in range(width):
            left = max(0, w - 1)
            right = min(width - 1, w + 1)
            top = max(0, h - 1)

            # Calculate costs for three possible seams and select the minimum
            cost_a = cumulative_energy[top, left] + np.abs(gradient_x[h, left] - gradient_x[h, right]) + np.abs(gradient_x[h, left] - gradient_x[top, w])
            cost_b = cumulative_energy[top, w] + np.abs(gradient_x[h, left] - gradient_x[h, right])
            cost_c = cumulative_energy[top, right] + np.abs(gradient_x[h, left] - gradient_x[h, right]) + np.abs(gradient_x[h, right] - gradient_x[top, w])

            cumulative_energy[h, w] += min(cost_a, cost_b, cost_c)

    return cumulative_energy

@jit
def backward_energy_matrix(image):
    # Compute the backward energy matrix using Sobel operators
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gradient_x = cv2.Sobel(src=gray_image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    gradient_y = cv2.Sobel(src=gray_image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    energy_map = np.abs(gradient_x) + np.abs(gradient_y)

    height, width = energy_map.shape
    cumulative_energy = np.copy(energy_map)

    for h in range(1, height):
        for w in range(width):
            left = max(0, w - 1)
            right = min(width, w + 2)
            cumulative_energy[h, w] += min(cumulative_energy[h - 1, left:right])

    return cumulative_energy

@jit
def vertical_seam_indices(cost_matrix):
    # Find the vertical seam indices using dynamic programming
    height, width = cost_matrix.shape
    seam_indices = np.zeros(height, dtype=np.int32)

    # Find the minimum cost index in the last row
    seam_indices[-1] = np.argmin(cost_matrix[-1])

    for h in range(height - 2, -1, -1):
        prev_index = seam_indices[h + 1]
        # Find the range for the argmin operation
        start = max(0, prev_index - 1)
        end = min(width, prev_index + 2)
        seam_indices[h] = start + np.argmin(cost_matrix[h, start:end])

    return seam_indices.tolist()

@jit
def remove_seam(image, seam_indices):
    # Remove the seam from the image
    height, width = image.shape[:2]
    output_width = width - 1

    output = np.zeros((height, output_width) + image.shape[2:], dtype=image.dtype)

    for h in range(height):
        current_index = seam_indices[h]
        if current_index > 0:
            output[h, :current_index] = image[h, :current_index]
        if current_index < width - 1:
            output[h, current_index:] = image[h, current_index + 1:]

    return output

@jit
def image_seam_removal(image, seams, cost_function, red_seams=False):
    # Remove seams from the image
    seam_removal_image = np.copy(image)

    for seam in range(seams):
        cost_matrix = cost_function(seam_removal_image)
        seam_indices = vertical_seam_indices(cost_matrix)
        seam_removal_image = remove_seam(seam_removal_image, seam_indices)

    return seam_removal_image



##### Horizontal functions 
@jit
def forward_energy_horizontal(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    rows, cols = gray_image.shape

    # Initialize cumulative energy matrix
    cumulative_energy = np.zeros((rows, cols))

    # Calculate energy differences for neighboring pixels
    energy_left_right = np.abs(np.roll(gray_image, -1, axis=1) - gray_image)
    energy_up_left_right = energy_left_right + np.abs(np.roll(gray_image, -1, axis=0) - np.roll(gray_image, -1, axis=1))
    energy_up_right = energy_left_right + np.abs(np.roll(gray_image, -1, axis=0) - np.roll(gray_image, 1, axis=1))

    # Initialize cumulative energy for the first column
    cumulative_energy[:, 0] = np.copy(gray_image[:, 0])

    # Calculate cumulative energy without considering boundaries
    for w in range(1, cols):
        for h in range(rows):
            # Define ranges for neighbor pixel indices
            h_up = max(h - 1, 0)
            h_down = min(h + 1, rows - 1)

            # Calculate energy for three possible seams
            options = [cumulative_energy[h_up, w - 1] + energy_up_left_right[h, w],
                       cumulative_energy[h, w - 1] + energy_left_right[h, w],
                       cumulative_energy[h_down, w - 1] + energy_up_right[h, w]]

            # Find the minimum cumulative energy among the options
            cumulative_energy[h, w] = gray_image[h, w] + min(options)

    return cumulative_energy


@jit
def backward_energy_horizontal(image):
    # Compute the backward energy matrix using Sobel operators (for horizontal seams)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gradient_x = cv2.Sobel(src=gray_image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    gradient_y = cv2.Sobel(src=gray_image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    energy_map = np.abs(gradient_x) + np.abs(gradient_y)

    cumulative_energy = np.copy(energy_map)
    height, width = cumulative_energy.shape

    for w in range(1, width):
        top = np.clip(np.arange(height) - 1, 0, height - 1)
        bottom = np.clip(np.arange(height) + 1, 0, height - 1)

        # Find minimum values among the three neighboring pixels in the previous column
        min_neighbors = np.minimum.reduce([cumulative_energy[top, w - 1], 
                                           cumulative_energy[:, w - 1], 
                                           cumulative_energy[bottom, w - 1]])

        cumulative_energy[:, w] += min_neighbors

    return cumulative_energy



@jit(nopython=True)
def horizontal_seam_indices(cost_matrix):
    height, width = cost_matrix.shape
    seam_indices = [np.argmin(cost_matrix[:, -1])]

    for w in range(width - 2, -1, -1):
        current_index = seam_indices[-1]
        start = max(0, current_index - 1)
        end = min(height, current_index + 2)
        seam_indices.append(start + np.argmin(cost_matrix[start:end, w]))

    seam_indices.reverse()
    return seam_indices


@jit
def remove_horizontal_seam(image, seam_indices):
    # Remove the horizontal seam from the image
    if len(image.shape) == 3:
        output = np.zeros([image.shape[0] - 1, image.shape[1], image.shape[2]], dtype=image.dtype)
    else:
        output = np.zeros([image.shape[0] - 1, image.shape[1]], dtype=image.dtype)

    for w in range(image.shape[1]):
        current_index = seam_indices[w]
        output[:, w] = np.concatenate((image[:current_index, w], image[current_index + 1:, w]), axis=0)

    return output

@jit
def image_seam_removal_horizontal(image, seams, cost_function, red_seams=False):
    seam_removal_image = np.copy(image)

    for seam in range(seams):
        cost_matrix = cost_function(seam_removal_image)
        seam_indices = horizontal_seam_indices(cost_matrix)
        seam_removal_image = remove_horizontal_seam(seam_removal_image, seam_indices)

    return seam_removal_image



@jit
def monte_carlo_seam_carving(image, k1, k2, iterations=5):
    best_sequence = None
    best_energy = float('inf')
    for _ in range(iterations):
        current_image = image.copy()
        sequence = ['row' for _ in range(k2)] + ['column' for _ in range(k1)]
        random.shuffle(sequence)

        for action in sequence:
            if action == 'row':
                seam = horizontal_seam_indices(forward_energy_horizontal(current_image))
                current_image = remove_horizontal_seam(current_image, seam)
            else:
                seam = vertical_seam_indices(forward_energy_matrix(current_image))
                current_image = remove_seam(current_image, seam)

        # Here we assume that the energy introduced by each seam removal can be summed up.
        current_energy = np.sum(forward_energy_matrix(current_image)) + np.sum(forward_energy_horizontal(current_image))
        if current_energy < best_energy:
            best_energy = current_energy
            best_sequence = sequence

    return best_sequence, best_energy


def get_image_from_sequence(sequence, image):
    for action in sequence:
        if action == 'row':
            seam = horizontal_seam_indices(forward_energy_horizontal(image))
            image = remove_horizontal_seam(image, seam)
        else:
            seam = vertical_seam_indices(forward_energy_matrix(image))
            image = remove_seam(image, seam)

    return image

if __name__ == "__main__":
    start = time.time()
    image_path = "Image1.bmp"
    image = cv2.imread(image_path)

    # Get user input for the number of columns and rows to remove
    columns_to_remove = int(input("Enter the number of columns to remove: "))
    rows_to_remove = int(input("Enter the number of rows to remove: "))

    # Perform Monte Carlo seam removal
    best_sequence, best_energy = monte_carlo_seam_carving(image, columns_to_remove, rows_to_remove, iterations=5)
    print("Best sequence found:", best_sequence)
    print("Best energy found:", best_energy)

    # Apply the best sequence to the original image
    image = get_image_from_sequence(best_sequence, image)
    cv2.imwrite("result_monte_carlo.bmp", image)
    end = time.time()
    print("With jit Monte Carlo adaptive sem removal time:", end-start)
