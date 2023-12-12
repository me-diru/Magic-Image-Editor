
#adaptive removal
import time
import numpy as np
import cv2
from numba import jit
import warnings
import numba
warnings.filterwarnings('ignore', category= numba.NumbaWarning)

@jit
def forward_energy_matrix(image):
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

@jit(nopython=True)
def vertical_seam_indices(cost_matrix):
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
def remove_seam(image, seam_indices):
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


@jit(nopython=True)
def image_seam_removal(image, seams, cost_function):
    # Remove seams from the image
    seam_removal_image = np.copy(image)
    index_tracker = np.array([np.arange(image.shape[1])] * image.shape[0])

    for seam in range(seams):
        cost_matrix = cost_function(seam_removal_image)
        seam_indices = vertical_seam_indices(cost_matrix)
        seam_removal_image = remove_seam(seam_removal_image, seam_indices)

    
    return seam_removal_image

#####
# horizontal functions
@jit
def forward_energy_matrix_horizontal(image):
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



@jit(nopython=True)
def vertical_seam_indices_horizontal(cost_matrix):
    # Find the horizontal seam indices using dynamic programming
    height, width = cost_matrix.shape
    seam_indices = [np.argmin(cost_matrix[:, width - 1])]

    for w in range(width - 2, -1, -1):
        current_index = seam_indices[-1]
        if current_index == 0:
            seam_indices.append(np.argmin(cost_matrix[0:2, w]))
        elif current_index == height - 1:
            seam_indices.append(height - 2 + np.argmin(cost_matrix[height - 2:height, w]))
        else:
            seam_indices.append(current_index - 1 + np.argmin(cost_matrix[current_index - 1:current_index + 2, w]))

    seam_indices.reverse()
    return seam_indices

@jit(nopython=True)
def paint_seam_horizontal(image, seam_indices, index_tracker):
    # Highlight the horizontal seam in the image by changing pixel color
    for w in range(image.shape[1]):
        current_index = min(max(index_tracker[seam_indices[w], w], 0), image.shape[0] - 1)
        image[current_index, w] = [0, 0, 255]

@jit
def remove_seam_horizontal(image, seam_indices):
    # Remove the horizontal seam from the image
    if len(image.shape) == 3:
        output = np.zeros([image.shape[0] - 1, image.shape[1], image.shape[2]], dtype=image.dtype)
    else:
        output = np.zeros([image.shape[0] - 1, image.shape[1]], dtype=image.dtype)

    for w in range(image.shape[1]):
        current_index = seam_indices[w]
        output[:current_index, w] = image[:current_index, w]
        output[current_index:, w] = image[current_index + 1:, w]

    return output

@jit(nopython=True)
def image_seam_removal_horizontal(image, seams, cost_function, red_seams=False):
    # Remove horizontal seams from the image
    seam_removal_image = np.copy(image)
    red_seam_image = np.copy(image)
    index_tracker = np.array([np.arange(image.shape[1])] * image.shape[0])

    for seam in range(seams):
        cost_matrix = cost_function(seam_removal_image)
        seam_indices = vertical_seam_indices_horizontal(cost_matrix)
        seam_removal_image = remove_seam_horizontal(seam_removal_image, seam_indices)

        if red_seams:
            paint_seam_horizontal(red_seam_image, seam_indices, index_tracker)
            index_tracker = remove_seam_horizontal(index_tracker, seam_indices)

    return seam_removal_image, red_seam_image

@jit(nopython=True)
def image_seam_removal_forward_horizontal(image, seams, cost_function, red_seams=False):
    # Remove horizontal seams using forward energy calculation
    seam_removal_image = np.copy(image)
    red_seam_image = np.copy(image)
    index_tracker = np.array([np.arange(image.shape[1])] * image.shape[0])

    for seam in range(seams):
        cost_matrix = cost_function(seam_removal_image)
        seam_indices = vertical_seam_indices_horizontal(cost_matrix)
        seam_removal_image = remove_seam_horizontal(seam_removal_image, seam_indices)

        if red_seams:
            paint_seam_horizontal(red_seam_image, seam_indices, index_tracker)
            index_tracker = remove_seam_horizontal(index_tracker, seam_indices)

    return seam_removal_image, red_seam_image


@jit
def adaptive_seam_removal(image, k1, k2):
    while k1 > 0 or k2 > 0:
        if k1 > 0:
            vertical_energy = np.sum(forward_energy_matrix(image))
        else:
            vertical_energy = float('inf')

        if k2 > 0:
            horizontal_energy = np.sum(forward_energy_matrix_horizontal(image))
        else:
            horizontal_energy = float('inf')

        if vertical_energy < horizontal_energy:
            # Remove a vertical seam
            seam_indices = vertical_seam_indices(forward_energy_matrix(image))
            image = remove_seam(image, seam_indices)
            k1 -= 1
        else:
            # Remove a horizontal seam
            seam_indices = vertical_seam_indices_horizontal(forward_energy_matrix_horizontal(image))
            image = remove_seam_horizontal(image, seam_indices)
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
    cv2.imwrite("result_combined-jit.bmp", result)

    end = time.time()
    print("With jit adaptive sem removal time:", end-start)
