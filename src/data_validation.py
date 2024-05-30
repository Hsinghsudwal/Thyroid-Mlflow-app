import pandas as pd
import os
import sys
from src.logger import logging
from scipy.stats import ks_2samp
from src.utility_file import Utility
from src.yaml_file import read_yaml_file, write_yaml_file

params = Utility().read_params()
# yml=Utility().read_yaml_file()


class DataValidation:

    def __init__(self) -> None:
        pass

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise e
        
    def columns_validate(self):
        try:
            report = {}
            threshold = 0.05

            logging.info("Loading schema file for data validation")
            SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

            schema_config = read_yaml_file(SCHEMA_FILE_PATH)

            number_of_columns = len(schema_config["columns"])
            print(number_of_columns)

            # reading paths
            artifact_dir = params["DATA_LOCATION"]["DATA_ARTIFACTS"]
            metric_path_folder = params["METRICS_PATH"]["METRICS_FOLDER"]
            raw_data_path = params["DATA_LOCATION"]["RAW_FILE_NAME"]
            train_data_path = params["DATA_LOCATION"]["TRAIN_FILE_NAME"]
            test_data_path = params["DATA_LOCATION"]["TEST_FILE_NAME"]

            # reading data artifacts
            raw_data = DataValidation.read_data(
                os.path.join(artifact_dir, str(raw_data_path))
            )
            train_data = DataValidation.read_data(
                os.path.join(artifact_dir, str(train_data_path))
            )
            test_data = DataValidation.read_data(
                os.path.join(artifact_dir, str(test_data_path))
            )

            logging.info(f"Schema files number of columns: {number_of_columns}")
            logging.info(f"Data frame columns for raw file: {len(raw_data.columns)}")
            logging.info(f"Data frame columns for train file: {len(train_data.columns)}")
            logging.info(f"Data frame columns for test file: {len(test_data.columns)}")

            if (len(raw_data.columns) & len(train_data.columns) & len(test_data.columns) == number_of_columns):
                is_found = False
                train_path = train_data.columns
                test_path = test_data.columns
                train_test_set = ks_2samp(train_path, test_path)

                logging.info(f"{train_test_set}")
                # print(train_test_set.pvalue)
                if threshold <= train_test_set.pvalue:
                    report.update(
                        {
                            "p-value": float(train_test_set.pvalue),
                            "dataset_drift": is_found
                        }
                    )

                logging.info("Output drift report for the train and test data")
                report_name = params["METRICS_PATH"]["DATA_VALIDATION_DRIFT_REPORT_FILE_NAME"]

                Utility().create_folder(metric_path_folder)
                drift_report_file_path = os.path.join(metric_path_folder, str(report_name))

                write_yaml_file(drift_report_file_path,report)
                logging.info("Output metics-> report.json.")

            else:
                logging.error("Error has occurred due to data drift")

        except Exception as e:
            raise e
        
