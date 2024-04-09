import pandas as pd
import numpy as np
from logger import logging
import os
from utility_file import Utility
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocess import Preprocessor
from src.model_trainer import Model
# import aws

param = Utility().read_params()

class MakeDataset:

    def __init__(self) -> None:
        pass

    def load_and_save(self):
        """This method is used to read the data from aws s3 storage and save to local drive.

        Returns
        --------
        None
        """

        try:
            logging.info('Loading of data from the source has started.')
            #data_url = params['data_location']['notebook_data']
            #raw_data_filename = params['data_location']['raw_data_filename']

            url = param['data_location']['notebook_data']
            raw_data_filename = param['data_location']['raw_data_filename']


            # getting data url from params.yaml file
            url = param['data_location']['notebook_data']

            logging.info("reading csv to dataset")
            # Reading the csv file

            data = pd.read_csv(url)

            main_data_folder = param['data_location']['data_artifact']

            # Creating a Data folder to save the loaded data
            Utility().create_folder(main_data_folder)

            # Saving the loaded data to the Data folder
            data.to_csv(os.path.join(main_data_folder, str(
                raw_data_filename)), index=False, sep=',')
            logging.info("File has saved to artifact folder ")

        except Exception as e:
            logging.error(e)
            raise e
