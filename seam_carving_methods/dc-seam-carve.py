#adaptive removal
import numpy as np
import cv2
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
            seam_indices = find_horizontal_seam(compute_backward_energy_horizontal(image))
            image = remove_horizontal_seam(image, seam_indices)
            k2 -= 1

    return image


@jit
def divide_image(image, num_segments, is_horizontal):
    segments = []
    if is_horizontal:
        segment_height = image.shape[0] // num_segments
        for i in range(num_segments):
            start_row = i * segment_height
            end_row = (i + 1) * segment_height if i < num_segments - 1 else image.shape[0]
            segments.append(image[start_row:end_row, :])
    else:
        segment_width = image.shape[1] // num_segments
        for i in range(num_segments):
            start_col = i * segment_width
            end_col = (i + 1) * segment_width if i < num_segments - 1 else image.shape[1]
            segments.append(image[:, start_col:end_col])
    return segments

@jit
def combine_segments(segments, is_horizontal):
    if is_horizontal:
        return np.concatenate(segments, axis=0)
    else:
        return np.concatenate(segments, axis=1)

@jit
def adaptive_seam_removal_divide_and_conquer(image, k1, k2, num_segments):
    horizontal_segments = divide_image(image, num_segments, True)
    vertical_segments = divide_image(image, num_segments, True)

    for i in range(num_segments):
        horizontal_segments[i] = adaptive_seam_removal(horizontal_segments[i], k1 // num_segments, k2 // num_segments)
        vertical_segments[i] = adaptive_seam_removal(vertical_segments[i], k1 // num_segments, k2 // num_segments)

    combined_horizontal = combine_segments(horizontal_segments, True)
    combined_vertical = combine_segments(vertical_segments, False)

    return combined_horizontal 

# Main execution
if __name__ == "__main__":
    image_path = "Image1.bmp"
    image = cv2.imread(image_path)

    columns_to_remove = int(input("Enter the number of columns to remove: "))
    rows_to_remove = int(input("Enter the number of rows to remove: "))
    num_segments = 2  # Example number of segments to test

    result = adaptive_seam_removal_divide_and_conquer(image, columns_to_remove, rows_to_remove, num_segments)
    cv2.imwrite("Images/divide_and_conquer_result.bmp", result)
