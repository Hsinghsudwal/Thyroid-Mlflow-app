import yaml
import os


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