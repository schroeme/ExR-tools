"""
Sets up the project configration. 
"""
import os
import json
import pathlib
from typing import List, Optional
from exr.io import create_folder_structure
from exr.utils import chmod, configure_logger

logger = configure_logger('ExR-Tools')

class Config:
    r"""
    A class used to manage and represent the configuration for the ExR-Tools software.

    This class handles the setup of paths, parameters, and project-specific settings. It also manages 
    the creation of directory structures and file permissions as required. Configurations can be saved to 
    or loaded from a JSON file for persistent use.

    **Methods:**
        - `set_config`: Configures various parameters for the ExR-Tools.
        - `save_config`: Saves the current configuration to a JSON file.
        - `load_config`: Loads configuration from a JSON file.
        - `create_directory_structure`: Creates the necessary directory structure for the project.
        - `set_permissions`: Modifies file permissions for the project directory.
        - `print`: Outputs all current configuration attributes.
    """
    
    def __init__(self):
        pass

    def __str__(self):
        r"""Returns a string representation of the Config object."""
        return str(self.__dict__)
    
    def __repr__(self):
        r"""Returns a string that reproduces the Config object when fed to eval()."""
        return self.__str__()

    def set_config(self,
                   raw_data_path: str,
                   processed_data_path: Optional[str] = None,
                   rounds: List[int] = list(range(10)),
                   rois: Optional[int] = None, 
                   spacing: List[float] = [0.250,0.1625,0.1625],
                   channel_names: List[str] = ['640','561','488'],
                   ref_round: int = 1,
                   ref_channel: str = '640',
                   permission: Optional[bool] = False,
                   create_directory_structure: Optional[bool] = True,
                   config_file_name: str = 'exr_tools_config',
                  ) -> None:
        r"""
        Sets up the configuration parameters for running ExR-Tools.

        :param raw_data_path: The absolute path to the raw data directory. There is no default value, this must be provided.
        :type raw_data_path: str
        :param processed_data_path: The absolute path to the processed data directory. Default is a 'processed_data' subdirectory inside the raw_data_path.
        :type processed_data_path: str
        :param rounds: List of the number of rounds in the ExR dataset. Default is list(range(10)).
        :type rounds: List[int]
        :param rois: The number of region of interests. Default is None, in which case the regions of interest will be inferred from the raw data.
        :type rois: List[int]
        :param spacing: Spacing between pixels in the format [X,Y,Z]. Default is [0.1625,0.1625,0.250].
        :type spacing: List[float]
        :param channel_names: Names of channels in the ND2 file. Default is ['633','546','488'].
        :type channel_names: List[str]
        :param ref_round: Reference round number. Default is 1.
        :type ref_round: int
        :param ref_channel: Reference channel name. Default is '633'.
        :type ref_channel: str
        :param permission: If set to True, changes permission of the raw_data_path to allow other users to read and write on the generated files. Default is None, in which case permissions will not be modified.
        :type permission: bool
        :param create_directroy_struc: If set to True, creates the directory structure in the specified project path. Default is None, in which case the directory structure will not be created.
        :type create_directroy_struc: bool
        :param config_file_name: Name of the configuration file. Default is 'exr_tools_config'.
        :type config_file_name: str
        """
        try:
            self.raw_data_path = os.path.abspath(raw_data_path)
            self.rounds = rounds
            self.spacing = spacing
            self.ref_round = ref_round
            self.ref_channel = ref_channel
            self.channel_names = channel_names

            # Input ND2 path
            self.nd2_path = os.path.join(self.raw_data_path , "R{}" , "ROI{}.nd2")

            #TODO start files name at 0        
            if rois is None:
                self.rois = list(range(1,len(os.listdir(os.path.dirname(self.nd2_path.format(self.ref_round,1))))+1))
            else:
                self.rois = list(range(1,rois+1))

            # Output h5 path
            if processed_data_path is not None:
                self.processed_data_path = os.path.abspath(processed_data_path)
            else:
                self.processed_data_path = os.path.join(self.raw_data_path,"processed_data")

            self.h5_path = os.path.join(self.processed_data_path , "R{}" , "ROI{}.h5")
            self.roi_analysis_path = os.path.join(self.processed_data_path,"roi_analysis")

            if create_directory_structure is not None:
                self.create_directory_structure()

            if permission is not None:
                self.set_permissions()

            self.save_config(config_file_name)

        except Exception as e:
            logger.error(f"Failed to set configuration. Error: {e}")
            raise

    def create_directory_structure(self):
        r"""Creates the directory structure in the specified project path."""
        try:
            create_folder_structure(str(self.processed_data_path),self.rois, self.rounds)
        except Exception as e:
            logger.error(f"Failed to create directory structure. Error: {e}")
            raise

    def set_permissions(self):
        r"""Changes permission of the raw_data_path to allow other users to read and write on the generated files."""
        try:
            chmod(pathlib.Path(self.processed_data_path))
        except Exception as e:
            logger.error(f"Failed to set permissions. Error: {e}")
            raise

    def save_config(self, config_file_name):
        r"""Saves the configuration to a .json file.

        :param config_file_name: Name of the configuration file.
        :type config_file_name: str
        """
        try:
            with open(os.path.join(self.processed_data_path , config_file_name + '.json'), "w") as f:
                json.dump(self.__dict__, f)
        except Exception as e:
            logger.error(f"Failed to save configuration. Error: {e}")
            raise

    def load_config(self, param_path: str) -> None:
        r"""Saves the configuration to a .json file.

        :param config_file_name: Name of the configuration file.
        :type config_file_name: str
        """
        try:
            param_path = os.path.abspath(param_path)
            with open(param_path, "r") as f:
                self.__dict__.update(json.load(f))
        except Exception as e:
            logger.error(f"Failed to load configuration. Error: {e}")
            raise
    
    def print(self) -> None:
        r"""Prints all attributes of the Config object."""
        try:
            for key, value in self.__dict__.items():
                print(f"{key}: {value}")
        except Exception as e:
            logger.error(f"Failed to print configuration. Error: {e}")
            raise
