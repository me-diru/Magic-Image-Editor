import time
import numpy as np
import cv2

def compute_forward_energy_horizontal(image):
    """ Compute the forward energy for horizontal seam carving. """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    rows, cols = gray_image.shape

    cumulative_energy = np.zeros((rows, cols))
    energy_diff_lr = np.abs(np.roll(gray_image, -1, axis=1) - gray_image)
    energy_diff_ulr = energy_diff_lr + np.abs(np.roll(gray_image, -1, axis=0) - np.roll(gray_image, -1, axis=1))
    energy_diff_ur = energy_diff_lr + np.abs(np.roll(gray_image, -1, axis=0) - np.roll(gray_image, 1, axis=1))

    cumulative_energy[:, 0] = gray_image[:, 0]

    for col in range(1, cols):
        for row in range(rows):
            upper = max(row - 1, 0)
            lower = min(row + 1, rows - 1)
            options = [
                cumulative_energy[upper, col - 1] + energy_diff_ulr[row, col],
                cumulative_energy[row, col - 1] + energy_diff_lr[row, col],
                cumulative_energy[lower, col - 1] + energy_diff_ur[row, col]
            ]
            cumulative_energy[row, col] = gray_image[row, col] + min(options)

    return cumulative_energy

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

def remove_horizontal_seams(image, num_seams, energy_func):
    """ Remove horizontal seams from an image. """
    for _ in range(num_seams):
        energy_matrix = energy_func(image)
        seam = find_horizontal_seam(energy_matrix)
        image = remove_horizontal_seam(image, seam)

    return image

def main():
    start = time.time()
    image_path = "Image1.bmp"
    seams_to_remove = int(input("Enter the number of rows to remove: "))

    image = cv2.imread(image_path)
    result = remove_horizontal_seams(image, seams_to_remove, compute_forward_energy_horizontal)

    cv2.imwrite("result_backward_horizontal.bmp", result)

    end = time.time()

    print(end-start)
if __name__ == "__main__":
    main()
