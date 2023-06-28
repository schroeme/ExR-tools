"""
Sets up the project parameters. 
"""
import os
import pickle

from nd2reader import ND2Reader
from exr.io import createfolderstruc

from exr.utils import configure_logger
logger = configure_logger('ExSeq-Toolbox')


class Config:
    def __init__(self):
        pass

    def set_config(self,
                project_path = '',
                codes = list(range(10)),
                fovs = None,
                ref_code = 0,
                thresholds = [200,300,300,200],
                spacing = [1.625,1.625,4.0],
                gene_digit_csv = 'gene_list.csv', #'/mp/nas3/ruihan/20230308_celegans/code0/gene_list.csv'
                permission = False,
                create_directroy_struc = False,
                args_file_name = None,
                ):
        
        self.project_path = project_path
        self.codes = codes
        self.thresholds = thresholds
        self.spacing = spacing
        self.permission = permission
        self.ref_code = ref_code
        
        # Housekeeping
        self.code2num = {'a':'0','c':'1','g':'2','t':'3'}
        self.colors = ['red','yellow','green','blue']
        self.colorscales = ['Reds','Oranges','Greens','Blues']
        self.channel_names = ['640','594','561','488','405']
        
        # Input ND2 path
        self.nd2_path = os.path.join(
            self.project_path, "code{}/Channel{} SD_Seq000{}.nd2"
        )
        if not fovs and "fovs" not in dir(self):
            self.fovs = list(
                ND2Reader(self.nd2_path.format(self.ref_code, "405", 4)).metadata[
                    "fields_of_view"
                ]
            )
        else:
            self.fovs = fovs

        # Output h5 path
        self.processed_path = os.path.join(self.project_path, "processed_ruihan")
        self.h5_path = os.path.join(self.processed_path, "code{}/{}.h5")
        self.tform_path = os.path.join(self.processed_path, "code{}/tforms/{}.txt")

        # Housekeeping
        self.code2num = {"a": "0", "c": "1", "g": "2", "t": "3"}
        self.colors = ["red", "yellow", "green", "blue"]
        self.colorscales = ["Reds", "Oranges", "Greens", "Blues"]
        self.channel_names = ["640", "594", "561", "488", "405"]

        self.work_path = self.project_path + "puncta/"

        self.gene_digit_csv = gene_digit_csv

        if create_directroy_struc:
            createfolderstruc(project_path, codes)

        if args_file_name == None:
            args_file_name = 'args.pkl'

        with open(os.path.join(self.project_path, args_file_name), "wb") as f:
            pickle.dump(self.__dict__, f)

        if permission:
            chmod(self.project_path)


    # load parameters from a pre-set .pkl file
    def load_config(self, param_path):
        r"""Loads and sets attributes from a .pkl file.

        :param str param_path: ``.pkl`` file path.
        """

        with open(os.path.abspath(param_path), "rb") as f:
            self.__dict__.update(pickle.load(f))




    