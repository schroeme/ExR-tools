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
