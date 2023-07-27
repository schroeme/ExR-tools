import h5py
import numpy as np
from exr.align.align_utils import alignment_NCC
from exr.utils import configure_logger
from exr.config import Config
from typing import List

logger = configure_logger('ExR-Tools')


def measure_round_alignment_NCC(config: Config, round: int, roi: int) -> List[float]:
    r"""
    Measures the alignment of a specific round and ROI (Region Of Interest) against a reference round using Normalized Cross-Correlation (NCC).

    :param config: Configuration options.
    :type config: Config
    :param round: The round to measure alignment for.
    :type round: int
    :param roi: The ROI to measure alignment for.
    :type roi: int
    :return: List of distance errors after alignment.
    :rtype: List[float]
    """
    distance_errors = []
    logger.info(
        f"Alignment Evaluation: Analyzing alignment between ref round:{config.ref_round} and round:{round} - ROI:{roi}")

    try:
        with h5py.File(config.h5_path.format(config.ref_round, roi), "r") as f:
            ref_vol = f[config.ref_channel][()]

        with h5py.File(config.h5_path.format(round, roi), "r") as f:
            aligned_vol = f[config.ref_channel][()]

        if np.count_nonzero(aligned_vol) > config.nonzero_thresh:
            ref_vol = (ref_vol - np.min(ref_vol)) / \
                (np.max(ref_vol) - np.min(ref_vol))
            aligned_vol = (aligned_vol - np.min(aligned_vol)) / \
                (np.max(aligned_vol) - np.min(aligned_vol))
            keepers = []

            for zz in range(aligned_vol.shape[0]):
                if np.count_nonzero(aligned_vol[zz, :, :]) > 0:
                    keepers.append(zz)

            logger.info(
                f"Alignment Evaluation: Round:{round} - ROI:{roi}, {len(keepers)} slices of {aligned_vol.shape[0]} kept.")

            if len(keepers) < 10:
                logger.info(
                    f"Alignment Evaluation: Round:{round} - ROI:{roi}, fewer than 10 slices. Skipping evaluation...")
            else:
                ref_vol = ref_vol[keepers, :, :]
                aligned_vol = aligned_vol[keepers, :, :]

                distance_errors = alignment_NCC(config, ref_vol, aligned_vol)

        return distance_errors

    except Exception as e:
        logger.error(
            f"Error during NCC alignment measurement for Round: {round}, ROI: {roi}, Error: {e}")
        raise
