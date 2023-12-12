import time
import numpy as np
import cv2
if __name__ == "__main__":
    from horizontal_seam_utilities import *

else:
    from .horizontal_seam_utilities import *




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

    cv2.imwrite("Images/horizontal_result.bmp", result)

    end = time.time()

    print(end-start)
if __name__ == "__main__":
    main()
