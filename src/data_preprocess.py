import pandas as pd
import numpy as np
from src.logger import logging
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from src.utility_file import Utility
from src.yaml_file import read_yaml_file, write_yaml_file

params = Utility().read_params()
Utility().create_folder("models")


class Preprocessor:

    def __init__(self) -> None:
        pass

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise e

    def data_validate(file_path):
        try:
            logging.info("Validating the data, before pre-process started")
            # report_path = read_yaml_file("metrics/report.yaml")
            report_path = read_yaml_file(file_path)
            # dict_list=list(report_path.keys())
            dict_list_value = list(report_path.values())
            # print(dict_list_value[0])
            status = False
            if dict_list_value[0] == status:
                logging.info("Data validation is completed")

        except Exception as e:
            logging.error(e)
            raise e


    def data_preprocess(self):
        try:
            logging.info("Preprocessing has started.")
            path_dir = "metrics/report.yaml"
            Preprocessor.data_validate(path_dir)

            # paths
            target = params["BASIC"]["TARGET_COLUMN"]
            artifact_folder = params["DATA_LOCATION"]["DATA_ARTIFACTS"]
            raw_path = params["DATA_LOCATION"]["RAW_FILE_NAME"]
            train_path = params["DATA_LOCATION"]["TRAIN_FILE_NAME"]
            test_path = params["DATA_LOCATION"]["TRAIN_FILE_NAME"]

            train_data = Preprocessor.read_data(
                os.path.join(artifact_folder, train_path)
            )
            test_data = Preprocessor.read_data(os.path.join(artifact_folder, test_path))

            #TRAINING
            train_data["age"] = np.where(train_data["age"] > 100, np.nan, train_data["age"])
            train_numerical_columns = train_data.select_dtypes(include=np.number)
            train_categorical_columns = train_data.select_dtypes(exclude="number")

            for i in train_numerical_columns:
                train_data[i].fillna(train_data[i].median(), inplace=True)

            for cat in train_categorical_columns:
                train_data[cat].fillna(method="ffill", inplace=True)

            logging.info("Processed Categorical and Numerical columns for training.")
            

            #TESTING
            test_data["age"] = np.where(test_data["age"] > 100, np.nan, test_data["age"])
            test_numerical_columns = test_data.select_dtypes(include=np.number)
            test_categorical_columns = test_data.select_dtypes(exclude="number")

            for i in test_numerical_columns:
                test_data[i].fillna(test_data[i].median(), inplace=True)

            for cat in test_categorical_columns:
                test_data[cat].fillna(method="ffill", inplace=True)

            logging.info("Processed Categorical and Numerical columns for testing.")


            # Saving the loaded data to the artificat folder for train
            Utility().create_folder(artifact_folder)
            filename = params["DATA_LOCATION"]["PREPROCESS_TRAIN_NAME"]
            # Saving the loaded data to the Data folder
            logging.info("saved csv file to the artifacts folder-> process_train_clean.csv file")
            train_data.to_csv(
                os.path.join(artifact_folder, str(filename)), index=False, sep=","
            )

            # Saving the loaded data to the artificat folder for test
            Utility().create_folder(artifact_folder)
            filename = params["DATA_LOCATION"]["PREPROCESS_TEST_NAME"]
            # Saving the loaded data to the Data folder
            logging.info("saved csv file to the artifacts folder-> process_test_clean.csv file")
            test_data.to_csv(
                os.path.join(artifact_folder, str(filename)), index=False, sep=","
            )
            logging.info("Preprocessing has completed")

            # using label encoder to normalize values for train
            train_df_input_feature = train_data.drop([target],axis=1)
            train_df_target_feature = train_data[target]
            test_df_input_feature = test_data.drop([target],axis=1)
            test_df_target_feature = test_data[target]

            le = LabelEncoder()
            for i in train_df_input_feature:
                try:
                    train_df_input_feature[i] = le.fit_transform(train_df_input_feature[i])
                except:
                    continue

            # using label encoder to normalize values for test
            le_test = LabelEncoder()
            for i in test_df_input_feature:
                try:
                    test_df_input_feature[i] = le_test.fit_transform(test_df_input_feature[i])
                except:
                    continue

            xtrain=train_df_input_feature
            xtest=train_df_target_feature
            ytrain=test_df_input_feature
            ytest=test_df_target_feature

            logging.info(f"{xtrain.shape, xtest.shape, ytrain.shape, ytest.shape}")
            logging.info("Saved preprocessing object.")

            # Utility().create_folder(model_dir)
            joblib.dump(le, "models\label_encoder.joblib")

            return (xtrain, ytrain, xtest, ytest)

        except Exception as e:
            logging.error(e)
            raise e

        