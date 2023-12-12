
#adaptive removal
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
def adaptive_seam_removal(image, k1, k2):
    while k1 > 0 or k2 > 0:
        if k1 > 0:
            vertical_energy = np.sum(compute_forward_energy_vertical(image))
        else:
            vertical_energy = float('inf')

        if k2 > 0:
            horizontal_energy = np.sum(compute_forward_energy_horizontal(image))
        else:
            horizontal_energy = float('inf')

        if vertical_energy < horizontal_energy:
            # Remove a vertical seam
            seam_indices = find_vertical_seam(compute_forward_energy_vertical(image))
            image = remove_vertical_seam(image, seam_indices)
            k1 -= 1
        else:
            # Remove a horizontal seam
            seam_indices = find_horizontal_seam(compute_forward_energy_horizontal(image))
            image = remove_horizontal_seam(image, seam_indices)
            k2 -= 1

    return image


if __name__ == "__main__":

    start = time.time()
    image_path = "Image1.bmp"
    image = cv2.imread(image_path)

    # Get user input for the number of columns and rows to remove
    columns_to_remove = int(input("Enter the number of columns to remove: "))
    rows_to_remove = int(input("Enter the number of rows to remove: "))

    # Perform adaptive seam removal
    result = adaptive_seam_removal(image, columns_to_remove, rows_to_remove)
    cv2.imwrite("Images/adaptive_greedy_combined_result.bmp", result)

    end = time.time()
    print("With jit adaptive sem removal time:", end-start)
