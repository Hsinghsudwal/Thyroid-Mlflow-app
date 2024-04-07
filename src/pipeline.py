import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_ingestion import MakeDataset
from src.data_preprocess import Preprocessor
from src.model_trainer import Model
from logger import logging

class Pipeline:

    def __init__(self) -> None:
        pass

    def make_pipeline():
        logging.info('Pileline has started')
        md = MakeDataset()
        md.load_and_save()
        logging.info('load and save')
        logging.info('preprocess has started')
        process=Preprocessor()
        process.preprocess_data()
        logging.info('model trainer has started')
        mt=Model()
        mt.model_training()

    if __name__ == "__main__":
        make_pipeline()

