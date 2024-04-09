import yaml
import os
import mlflow
from typing import Any

class Utility:

    def __init__(self, params_path='params.yaml') -> None:
        self.params_path = params_path

    def create_folder(self, folder_name):
        """This method is used to create folders that are required.

        Returns
        --------
        None
        """

        try:
            # Creating a directory if it does not exist already
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

        except Exception as e:
            raise e

    def read_params(self):
        """This method is used to read the parameters yaml file.

        Returns: the yaml file object.
        """
        try:
            # Reading params yaml file
            #params_path=os.path.dirname(os.path.abspath(params.yaml))
            #print(params_path)
            with open(self.params_path, 'r') as params_file:
                params = yaml.safe_load(params_file)

        except Exception as e:
            raise e

        else:
            return params


    def create_mlflow_experiment(
        experiment_name: str, artifact_location: str)->str:#, tags: dict[str, Any]) -> str:

        try:
            experiment_id = mlflow.create_experiment(
            name=experiment_name, artifact_location=artifact_location) #, tags=tags)
        except:
            print(f"Experiment {experiment_name} already exists.")
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

        mlflow.set_experiment(experiment_name=experiment_name)

        return experiment_id


    def get_mlflow_experiment(experiment_id: str = None, experiment_name: str = None
) -> mlflow.entities.Experiment:
        
        if experiment_id is not None:
         experiment = mlflow.get_experiment(experiment_id)
        elif experiment_name is not None:
            experiment = mlflow.get_experiment_by_name(experiment_name)
        else:
            raise ValueError("Either experiment_id or experiment_name must be provided.")
    
        return experiment