"""
Functions to assist in project directory creation. 
"""

import numpy as np
from pathlib import Path
from nd2reader import ND2Reader
from typing import Optional, List
from exr.utils import configure_logger
logger = configure_logger('ExR-Tools')


def createfolderstruc(processed_dir: Path, rounds: List[int]) -> None:
    """
    Creates a results folder for the specified code.

    :param processed_dir: The directory where all results for the specified code should be stored.
    :type processed_dir: Path
    :param rounds: The list of rounds to create the folder structure for.
    :type rounds: List[int]
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    for round in rounds:
        round_path = processed_dir / f"R{round}"
        round_path.mkdir(exist_ok=True)


def nd2ToVol(filename: str, channel_name: str = '640 SD', ratio: int = 1) -> Optional[np.ndarray]:
    """
    Generate a volume from ND2 file.

    :param filename: The name of the ND2 file.
    :type filename: str
    :param channel_name: The name of the channel, defaults to '633'
    :type channel_name: str, optional
    :param ratio: The ratio for downsampling, defaults to 1 (no downsampling)
    :type ratio: int, optional
    :return: A 3D numpy array representing the volume, if successful. None otherwise.
    :rtype: np.ndarray, optional

    """
    try:

        vol = ND2Reader(filename)
        channel_names = vol.metadata["channels"]
        channel_id = [x for x in range(len(channel_names)) if channel_name in channel_names[x]]
        assert len(channel_id) == 1
        channel_id = channel_id[0]

        out = np.zeros([len(vol), vol[0].shape[0]//ratio,
                       vol[0].shape[1] // ratio], np.uint16)

        for z in range(len(vol)):
            out[z] = vol.get_frame_2D(c=channel_id, t=0, z=z, x=0, y=0, v=0)[
                ::ratio, ::ratio]

        return out
    except ValueError:
        logger.error(f"Channel '{channel_name}' not found in the file.")
        return None
    except Exception as e:
        logger.error(f"Failed to generate volume from ND2 file. Error: {e}")
        return None
