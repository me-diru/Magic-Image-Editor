
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
    height, width, _ = image.shape
    dp = np.full((height+1, width+1), float('inf'))
    dp[0, 0] = 0

    # Populating the dynamic programming matrix
    for i in range(height):
        for j in range(width):
            if i < height - 1:
                dp[i+1, j] = min(dp[i+1, j], dp[i, j] + np.sum(compute_forward_energy_horizontal(image[i:i+2, :, :])))
            if j < width - 1:
                dp[i, j+1] = min(dp[i, j+1], dp[i, j] + np.sum(compute_forward_energy_vertical(image[:, j:j+2, :])))

    # Backtracking to find the optimal sequence of seams
    j, i = k1, k2
    while i > 0 and j > 0:
        if dp[i-1, j] < dp[i, j-1]:
            # Remove horizontal seam
            seam_indices = find_horizontal_seam(compute_backward_energy_horizontal(image))
            image = remove_horizontal_seam(image, seam_indices)
            i -= 1
        else:
            # Remove vertical seam
            seam_indices = find_vertical_seam(compute_forward_energy_vertical(image))
            image = remove_vertical_seam(image, seam_indices)
            j -= 1

    while i > 0:
        seam_indices = find_horizontal_seam(compute_backward_energy_horizontal(image))
        image = remove_horizontal_seam(image, seam_indices)
        i -= 1

    while j > 0:
        seam_indices = find_vertical_seam(compute_forward_energy_vertical(image))
        image = remove_vertical_seam(image, seam_indices)
        j -= 1

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
    cv2.imwrite("Images/dp_result.bmp", result)

    end = time.time()
    print("With jit DP adaptive sem removal time:", end-start)
