import pandas as pd
import os
import sys
from logger import logging


class Model:

    def __init__(self) -> None:
        pass

    def model_training(self):
        try:
            logging.info("Model Training initiate")
            pass
        except Exception as e:
            logging.error(e)
            raise e



