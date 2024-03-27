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

params = Utility().read_params()

class MakeDataset:

    def __init__(self) -> None:
        pass

    def load_and_save(self, url, filename):
        """This method is used to read the data from aws s3 storage and save to local drive.

        Returns
        --------
        None
        """

        try:
            # getting data url from params.yaml file
            url = params['data_location']['notebook_data']

            logging.info("reading csv to dataset")
            # Reading the csv file

            data = pd.read_csv(url)

            main_data_folder = params['data_location']['data_artifact']

            # Creating a Data folder to save the loaded data
            Utility().create_folder(main_data_folder)

            # Saving the loaded data to the Data folder
            data.to_csv(os.path.join(main_data_folder, str(
                filename)), index=False, sep=',')
            logging.info("File has saved to artifact folder ")

        except Exception as e:
            logging.error(e)
            raise e

def main():
    data_url = params['data_location']['notebook_data']
    raw_data_filename = params['data_location']['raw_data_filename']

    md = MakeDataset()
    logging.info('Loading of data from the source has started.')
    md.load_and_save(data_url, raw_data_filename)
    logging.info(
        'Data loading completed and data saved to the directory artifacts folder -> raw.csv file')

    process=Preprocessor()
    process.preprocess_data()
    mt=Model()
    mt.model_training()

if __name__ == "__main__":

    main()
    