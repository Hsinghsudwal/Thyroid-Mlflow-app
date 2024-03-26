from pathlib import Path
import os

def __init__(self, params_path='params.yaml') -> None:
        self.params_path = params_path

def read_params(self):
    params_path=os.path.dirname(os.path.abspath(__file__))
    print(params_path)