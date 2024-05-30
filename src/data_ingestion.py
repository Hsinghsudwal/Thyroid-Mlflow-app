import pandas as pd
import numpy as np
import os
from src.logger import logging
from src.utility_file import Utility
from sklearn.model_selection import train_test_split
import pymongo
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import aws

params = Utility().read_params()

class MakeDataset:

    def __init__(self) -> None:
        pass

    def load_and_save(self):

        try:
            logging.info('Loading of data from the source has started.')

            url= MongoClient(os.getenv('connection_to'))
            #database
            db=url['Thyroid-Database']
            # print(db)

            #collection
            collec=db['Thyroid-Collection']
            x=collec.find()
            newlist=[]
            for data in x:
                newlist.append(data)
            # print(newlist)

            df=pd.DataFrame(newlist)
            # print(df.head())

            # DATA_PATH
            artifacts_folder = params['DATA_LOCATION']['DATA_ARTIFACTS']
            Utility().create_folder(artifacts_folder)

            raw_file=params['DATA_LOCATION']['RAW_FILE_NAME']
            train_file=params['DATA_LOCATION']['TRAIN_FILE_NAME']
            test_file=params['DATA_LOCATION']['TEST_FILE_NAME']

            logging.info("Saving raw file to artifact folder -> raw.csv ")
            df.to_csv(os.path.join(artifacts_folder, str(
                raw_file)), index=False, sep=',')
            
            train_set,test_set=train_test_split(df,test_size=0.2, random_state=42)

            train_data_path=os.path.join(artifacts_folder,str(train_file))
            test_data_path=os.path.join(artifacts_folder,str(test_file))

            train_set.to_csv(train_data_path, index=False, header=True)
            test_set.to_csv(test_data_path, index=False, header=True)
            logging.info("Saving train file to artifact folder -> train.csv ")
            logging.info("Saving test file to artifact folder -> test.csv ")

            logging.info("Data Ingestion has completed")

            return(train_data_path,test_data_path)

        except Exception as e:
            logging.error(e)
            raise e
        

