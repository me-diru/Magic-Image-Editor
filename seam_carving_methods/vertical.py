#vertical removal
import numpy as np
import cv2
import time
if __name__ == "__main__":
    from vertical_seam_utilities import *
else:
    from .vertical_seam_utilities import *


def remove_vertical_seams(image, seams, cost_function):
    # Remove seams from the image
    seam_removal_image = np.copy(image)
    index_tracker = np.array([np.arange(image.shape[1])] * image.shape[0])

    for seam in range(seams):
        cost_matrix = cost_function(seam_removal_image)
        seam_indices = find_vertical_seam(cost_matrix)
        seam_removal_image = remove_vertical_seam(seam_removal_image, seam_indices)

    
    return seam_removal_image

if __name__ == "__main__":
    image = cv2.imread("Image1.bmp")
    start_time = time.time()  # Start time measurement

    
    # Get user input for the target width and height
    seams_to_remove = int(input("Enter the number of columns to remove: "))


    result = remove_vertical_seams(image, seams_to_remove, compute_forward_energy_vertical)


    end_time = time.time()

    print(f"Time taken for seam removal: {end_time - start_time} seconds")

    cv2.imwrite("Images/vertical_result.bmp", result)
   