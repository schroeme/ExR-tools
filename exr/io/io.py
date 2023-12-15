"""
The IO module in the ExR Tools is designed to streamline the process of reading, converting, and managing image data files for expansion microscopy
"""
import json
import numpy as np
from pathlib import Path
from nd2reader import ND2Reader
from typing import Optional, List
from exr.utils import configure_logger
logger = configure_logger('ExR-Tools')


def create_folder_structure(processed_dir: Path,rois:List[int], rounds: List[int]) -> None:
    r"""
    Creates a results folder for the specified code.

    :param processed_dir: The directory where all results for the specified code should be stored.
    :type processed_dir: Path
    :param rounds: The list of rounds to create the folder structure for.
    :type rounds: List[int]
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    roi_analysis_dir = processed_dir/ "roi_analysis"
    roi_analysis_dir.mkdir(parents=True, exist_ok=True)

    roi_analysis = ['segmentation_masks','synapses_properties','inspect_synapses']
    for dir in roi_analysis:
        roi_analysis_sub_dir = roi_analysis_dir / dir
        roi_analysis_sub_dir.mkdir(parents=True, exist_ok=True)
 
    for round in rounds:
        round_dir = processed_dir / f"R{round}"
        round_dir.mkdir(exist_ok=True)

    align_eval_dir = processed_dir / "alignment_evaluation"
    align_eval_dir.mkdir(parents=True, exist_ok=True)

    for roi in rois:
        roi_dir = align_eval_dir / f"ROI{roi}"
        roi_dir.mkdir(exist_ok=True)



def nd2ToVol(filename: str, channel_name: str = '640 SD', ratio: int = 1) -> Optional[np.ndarray]:
    r"""
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


def load_json(file_path):
    r"""
    Loads a JSON file from the specified path and returns the data.

    :param file_path: The path to the JSON file to load.
    :type file_path: str

    :return: The data loaded from the JSON file or None if an error occurs.
    :rtype: Union[Dict, None]
    """
    try:
        with open(file_path, 'r') as json_file:
            return json.load(json_file)
    except Exception as e:
        logger.error(f"Failed to load JSON from {file_path}: {e}")
        return None

def save_json(data, file_path):
    r"""
    Saves the provided data as a JSON file to the specified path.

    :param data: The data to save as a JSON file.
    :type data: Dict
    :param file_path: The path where the JSON file should be saved.
    :type file_path: str
    """
    try:
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
    except Exception as e:
        logger.error(f"Failed to save JSON to {file_path}: {e}")