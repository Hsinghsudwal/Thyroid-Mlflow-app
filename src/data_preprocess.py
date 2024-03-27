import pandas as pd
import numpy as np
from logger import logging
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from utility_file import Utility

params = Utility().read_params()
Utility().create_folder("models")


class Preprocessor:

    def __init__(self) -> None:
        pass

    def preprocess_data(self):

        try:
            logging.info("Preprocessing has started.")
            artifacts_path = params["data_location"]["data_artifact"]
            data_file_path = params["data_location"]["raw_data_filename"]
            DATA_PATH = os.path.join(artifacts_path, data_file_path)
            logging.info("Reading the data to perform preprocess")
            df = pd.read_csv(DATA_PATH)
            # print(df.head(5))
            df["age"] = np.where(df["age"] > 100, np.nan, df["age"])

            df = df.drop(
                [
                    "Unnamed: 0",
                    "TSH_measured",
                    "T3_measured",
                    "TT4_measured",
                    "T4U_measured",
                    "FTI_measured",
                    "TBG_measured",
                    "TBG",
                ],
                axis=1,
            )

            numerical_columns = df.select_dtypes(include=np.number)
            categorical_columns = df.select_dtypes(exclude="number")

            for i in numerical_columns:
                df[i].fillna(df[i].median(), inplace=True)

            for cat in categorical_columns:
                df[cat].fillna(method="ffill", inplace=True)

            logging.info("Processed Categorical and Numerical columns.")

            # Saving the loaded data to the artificat folder
            main_data_folder = params["data_location"]["data_artifact"]
            Utility().create_folder(main_data_folder)
            filename = params["data_location"]["preprocess_file"]
            # Saving the loaded data to the Data folder
            logging.info("saved csv file to the artifacts folder-> process.csv file")
            df.to_csv(
                os.path.join(main_data_folder, str(filename)), index=False, sep=","
            )
            logging.info("Preprocessing has completed")

            # Data Transformer
            logging.info("Columns Transformed has started")
            artifacts_path = params["data_location"]["data_artifact"]
            process_file = params["data_location"]["preprocess_file"]
            data_path_file = os.path.join(artifacts_path, process_file)
            data = pd.read_csv(data_path_file)

            # Spliting the data into dependent and independent
            X = data.drop(["classes"], axis=1)
            # y=data['classes']

            logging.info("Label Encoder Categorical Features.")
            # using label encoder to normalize values
            le = LabelEncoder()
            for i in X:
                try:
                    data[i] = le.fit_transform(data[i])
                except:
                    continue

            artifactFolder = params["data_location"]["data_artifact"]
            Utility().create_folder(artifactFolder)
            filename = params["data_location"]["transformer_file"]
            # Saving the loaded data to the Data folder
            data.to_csv(
                os.path.join(artifactFolder, str(filename)), index=False, sep=","
            )
            logging.info(
                "saved csv file to the artificats folder -> transformer.csv file."
            )
            # Save the label encoder using joblib for later use
            target = params["basic"]["target_column_name"]
            y = data[target]
            logging.info("Saving target label encoder to model folder -> model joblib.")
            label_encoder = LabelEncoder()
            target_encoded = label_encoder.fit_transform(y)
            joblib.dump(target_encoded, "models\label_encoder.joblib")
            logging.info("Columns transforming has finished")
            # print(data)

        except Exception as e:
            logging.error(e)
            raise e
