"""
Functions to assist in project directory creation. 
"""

import os
import numpy as np
from nd2reader import ND2Reader


from typing import Optional
from exr.utils import configure_logger
logger = configure_logger('ExR-Tools')


def createfolderstruc(processed_dir, rounds):
    r"""Creates a results folder for the specified code.

    :param str outdir: the directory where all results for the specified code should be stored.
    :param list codes: the list of codes to create the folder structure for.
    """

    if os.path.isdir(processed_dir) is False:
        os.makedirs(processed_dir)

    for round in rounds:

        round_path = os.path.join(processed_dir, "R{}".format(round))

        if os.path.isdir(round_path) is False:
            os.makedirs(round_path)




def nd2ToVol(filename: str, channel_name: str = '640 SD', ratio: int = 1) -> Optional[np.ndarray]:
    """
    Generate a volume from ND2 file.

    :param filename: The name of the ND2 file.
    :type filename: str
    :param channel_name: The name of the channel, defaults to '640 SD'
    :type channel_name: str, optional
    :param ratio: The ratio for downsampling, defaults to 1 (no downsampling)
    :type ratio: int, optional
    :return: A 3D numpy array representing the volume, if successful. None otherwise.
    :rtype: np.ndarray, optional

    :raises ValueError: If the channel_name is not found in the file
    """

    # Initialize ND2 reader
    vol = ND2Reader(filename)

    # Check if the desired channel exists in the file
    try:
        channel_id = vol.metadata['channels'].index(channel_name)
    except ValueError:
        logger.error(f"Channel '{channel_name}' not found in the file.")
        return None

    # Initialize output volume
    out = np.zeros([len(vol), vol[0].shape[0]//ratio , vol[0].shape[1] //ratio], np.uint16)

    # Populate the output volume
    for z in range(len(vol)):
        out[z] = vol.get_frame_2D(c=channel_id, t=0, z=z, x=0, y=0, v=0)[::ratio,::ratio]

    return out

