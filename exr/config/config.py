"""
Sets up the project parameters. 
"""
import os
import pickle
from exr.io import createfolderstruc
from exr.utils import chmod, configure_logger
logger = configure_logger('ExR-Tools')


class Config:
    def __init__(self):
        pass

    def set_config(self,
                raw_data_path,
                processed_data_path = None,
                rounds = list(range(10)),
                rois = None,
                ref_round = 1, # TODO fix
                spacing = [0.1625,0.1625,0.250],
                channel_names = ['633','546','488'],
                permission = False,
                create_directroy_struc = True,
                config_file_name = 'exr_tools_config',
                ):
        
        self.raw_data_path = os.path.abspath(raw_data_path)
        self.rounds = rounds
        self.spacing = spacing
        self.permission = permission
        self.ref_round = ref_round
        self.channel_names = channel_names
        
        # Input ND2 path
        self.nd2_path = os.path.join(
            self.raw_data_path, "R{}","40x ROI{}.nd2"
        )
        
        #TODO start files name at 0        
        if not rois and "rois" not in dir(self):
            self.rois = list(range(1,len(os.listdir(os.path.dirname(self.nd2_path.format(self.ref_round,1))))+1))
        else:
            self.rois = list(range(1,rois+1))

        # Output h5 path
        if processed_data_path:
            self.processed_data_path = os.path.abspath(processed_data_path)
        else:
            self.processed_data_path = os.path.join(self.raw_data_path, "processed_data")
        
        self.h5_path = os.path.join(self.processed_data_path, "R{}/{}.h5")

        # Housekeeping
        self.code2num = {"a": "0", "c": "1", "g": "2", "t": "3"}
        self.colors = ["red", "yellow", "green", "blue"]
        self.colorscales = ["Reds", "Oranges", "Greens", "Blues"]
        self.channel_names = ["640", "594", "561", "488", "405"]

        if create_directroy_struc:
            createfolderstruc(self.processed_data_path, self.rounds)

        with open(os.path.join(self.processed_data_path, config_file_name + '.pkl'), "wb") as f:
            pickle.dump(self.__dict__, f)

        if permission:
            chmod(self.raw_data_path)



    # load parameters from a pre-set .pkl file
    def load_config(self, param_path):
        r"""Loads and sets attributes from a .pkl file.

        :param str param_path: ``.pkl`` file path.
        """

        with open(os.path.abspath(param_path), "rb") as f:
            self.__dict__.update(pickle.load(f))

    def print(self):
        r"""Prints all attributes.
        """
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")



    