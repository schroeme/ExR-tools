"""
Functions to assist in project directory creation. 
"""

import os
import numpy as np
from nd2reader import ND2Reader


from typing import Optional
from exr.utils import configure_logger
logger = configure_logger('ExSeq-Toolbox')


def createfolderstruc(out_dir, codes):
    r"""Creates a results folder for the specified code.

    :param str outdir: the directory where all results for the specified code should be stored.
    :param list codes: the list of codes to create the folder structure for.
    """

    processed_dir = os.path.join(out_dir, "processed/")
    puncta_dir = os.path.join(out_dir, "puncta/")
    puncta_inspect_dir = os.path.join(puncta_dir, "inspect_puncta/")

    if os.path.isdir(processed_dir) is False:
        os.makedirs(processed_dir)

    if os.path.isdir(puncta_dir) is False:
        os.makedirs(puncta_dir)

    if os.path.isdir(puncta_inspect_dir) is False:
        os.makedirs(puncta_inspect_dir)

    for code in codes:

        code_path = os.path.join(processed_dir, "code{}".format(code))

        if os.path.isdir(code_path) is False:
            os.makedirs(code_path)

        tform_dir = os.path.join(code_path, "tforms")

        if os.path.isdir(tform_dir) is False:
            os.makedirs(tform_dir)



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

