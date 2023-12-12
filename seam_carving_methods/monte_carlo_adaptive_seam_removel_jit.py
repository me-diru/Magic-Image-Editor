
#adaptive removal
import random
import time
import numpy as np
import cv2
from numba import jit
if __name__ == "__main__":
    from horizontal_seam_utilities import *
    from vertical_seam_utilities import *
else:
    from .horizontal_seam_utilities import *
    from .vertical_seam_utilities import *


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
                seam = find_horizontal_seam(compute_forward_energy_horizontal(current_image))
                current_image = remove_horizontal_seam(current_image, seam)
            else:
                seam = find_vertical_seam(compute_forward_energy_vertical(current_image))
                current_image = remove_vertical_seam(current_image, seam)

        # Here we assume that the energy introduced by each seam removal can be summed up.
        current_energy = np.sum(compute_forward_energy_vertical(current_image)) + np.sum(compute_forward_energy_horizontal(current_image))
        if current_energy < best_energy:
            best_energy = current_energy
            best_sequence = sequence

    return best_sequence, best_energy


def get_image_from_sequence(sequence, image):
    for action in sequence:
        if action == 'row':
            seam = find_horizontal_seam(compute_forward_energy_horizontal(image))
            image = remove_horizontal_seam(image, seam)
        else:
            seam = find_vertical_seam(compute_backward_energy_vertical(image))
            image = remove_vertical_seam(image, seam)

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
    cv2.imwrite("Images/monte_carlo_result.bmp", image)
    end = time.time()
    print("With jit Monte Carlo adaptive sem removal time:", end-start)
