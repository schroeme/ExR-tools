import os
import numpy as np
from pathlib import Path
from typing import Optional
from multiprocessing import Pool
from skimage.restoration import rolling_ball
from scipy.ndimage import white_tophat
from skimage.morphology import disk
from exr.utils.log import configure_logger

logger = configure_logger('ExR-Tools')


def chmod(path: Path) -> None:
    """
    Sets permissions so that users and the owner can read, write and execute files at the given path.

    :param path: Path in which privileges should be granted.
    :type path: pathlib.Path
    """
    if os.name != "nt":  # Skip for Windows OS
        try:
            path.chmod(0o766)  # octal notation for permissions
        except Exception as e:
            logger.error(
                f"Failed to change permissions for {path}. Error: {e}")
            raise


def subtract_background_rolling_ball(volume: np.ndarray,
                                     radius: int = 50,
                                     num_threads: Optional[int] = 40) -> np.ndarray:
    """
    Performs background subtraction on a volume image using the rolling ball method.

    :param volume: The input volume image.
    :type volume: np.ndarray
    :param radius: The radius of the rolling ball used for background subtraction. Default is 50.
    :type radius: int, optional
    :param num_threads: The number of threads to use for the rolling ball operation. Default is 40.
    :type num_threads: int, optional
    :return: The volume image after background subtraction.
    :rtype: np.ndarray
    """
    corrected_volume = np.empty_like(volume)
    logger.info(f"Rolling_ball background subtraction")
    try:
        for slice_index in range(volume.shape[0]):
            corrected_volume[slice_index] = volume[slice_index] - rolling_ball(
                volume[slice_index], radius=radius, num_threads=num_threads)

        return corrected_volume
    except Exception as e:
        logger.error(f"Error during rolling ball background subtraction: {e}")
        raise


def subtract_background_top_hat(volume: np.ndarray,
                                radius: int = 50,
                                use_gpu: Optional[bool] = True) -> np.ndarray:
    """
    Performs top-hat background subtraction on a volume image.

    :param volume: The input volume image.
    :type volume: np.ndarray
    :param radius: The radius of the disk structuring element used for top-hat transformation. Default is 50.
    :type radius: int, optional
    :param use_gpu: If True, uses GPU for computation (requires cupy). Default is False.
    :type use_gpu: bool, optional
    :return: The volume image after background subtraction.
    :rtype: np.ndarray
    """
    structuring_element = disk(radius)
    corrected_volume = np.empty_like(volume)
    logger.info(f"top-hat background subtraction")
    try:
        if use_gpu:
            from cupyx.scipy.ndimage import white_tophat
            import cupy as cp

        for i in range(volume.shape[0]):
            if use_gpu:
                corrected_volume[i] = cp.asnumpy(
                    white_tophat(
                        cp.asarray(volume[i]),
                        structure=cp.asarray(structuring_element)
                    )
                )
            else:
                corrected_volume[i] = white_tophat(
                    volume[i], structure=structuring_element)

        return corrected_volume
    except Exception as e:
        logger.error(f"Error during top-hat background subtraction: {e}")
        raise



def calculate_volume_and_surface_area(labeled_image, label=1,spacing=[1,1,1]):
    """
    Calculate the volume and surface area of the object with the given label in the 3D labeled image.
    
    Parameters:
        labeled_image (numpy.ndarray): 3D array where different objects are labeled with different integer values.
        label (int): The label of the object for which to calculate volume and surface area.
    
    Returns:
        tuple: (volume, surface_area)
    """
    # Initialize volume and surface_area
    volume = 0
    surface_area = 0
    
    # Define the directions to check for neighboring voxels
    directions = [(0, 1, 0), (1, 0, 0), (0, -1, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)]
    
    # Iterate over the 3D array
    for z in range(labeled_image.shape[0]):
        for y in range(labeled_image.shape[1]):
            for x in range(labeled_image.shape[2]):
                # If this voxel is part of the object
                if labeled_image[z, y, x] == label:
                    # Increment the volume
                    volume += 1
                    
                    # Check the neighbors to calculate surface area
                    for dx, dy, dz in directions:
                        neighbor_x = x + dx
                        neighbor_y = y + dy
                        neighbor_z = z + dz
                        
                        # Check if the neighbor coordinates are within bounds
                        if 0 <= neighbor_x < labeled_image.shape[2] and \
                           0 <= neighbor_y < labeled_image.shape[1] and \
                           0 <= neighbor_z < labeled_image.shape[0]:
                            # If the neighbor is not part of the object, it's a boundary face
                            if labeled_image[neighbor_z, neighbor_y, neighbor_x] != label:
                                surface_area += 1
                        else:
                            # Edge of the image is also considered a boundary face
                            surface_area += 1
    
    z_dim , y_dim , x_dim = spacing
    physical_volume = volume * (z_dim * y_dim * x_dim)

    avg_face_area = (y_dim * x_dim + z_dim * y_dim + z_dim * x_dim) / 3
    # Calculate the physical surface area
    physical_surface_area = surface_area * avg_face_area

    return volume, surface_area, physical_volume, physical_surface_area











# multiprocessing acclerated
# def subtract_background_rolling_ball(volume_img, radius=50, num_threads=20):

#     def subtract_slice(args):
#         slice_index, volume_slice, radius, num_threads = args
#         logger.info(slice_index)
#         corrected_slice = volume_slice - rolling_ball(volume_slice, radius=radius, num_threads=num_threads)
#         return slice_index, corrected_slice

#     corrected_volume_img = np.empty_like(volume_img)

#     args = [(i, volume_img[i], radius, num_threads) for i in range(volume_img.shape[0])]

#     with Pool(processes=5) as pool:
#         results = pool.map(subtract_slice, args)

#     for slice_index, corrected_slice in sorted(results):
#         logger.info(slice_index)
#         corrected_volume_img[slice_index] = corrected_slice

#     return corrected_volume_img
