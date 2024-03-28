import pandas as pd 
import joblib
import pickle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from urllib.parse import urlparse
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import json
from logger import logging
import os

from utility_file import Utility
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

param = Utility().read_params()
Utility().create_folder('models')


class Model:
  
  def __init__(self) -> None:
      pass
    
  def model_training(self):
      try:
        logging.info('Model training on the data has started.')
        artifacts_path = param['data_location']['data_artifact']
        data_file_path=param['data_location']['transformer_file']
        DATA_PATH = os.path.join(artifacts_path,data_file_path)
        logging.info('Reading data to perform algorithms')
        data=pd.read_csv(DATA_PATH)
        #print(data.head(5))
        # Load label_encoder

        label_target=joblib.load('models/label_encoder.joblib')
        X=data.drop(['classes'],axis=1)
        y=label_target
        ros=RandomOverSampler()
        X_sampled,y_sampled = ros.fit_resample(X,y)
        #print(X_sampled.shape)
        X_sampled=pd.DataFrame(data=X_sampled,columns=X.columns)
        #print(X_sampled.head(5))
        # train test set
        RANDOM_STATE=param['basic']['random_state']
        TEST_SIZE=param['basic']['test_size']
        X_train,X_test,y_train,y_test=train_test_split(X_sampled,y_sampled,test_size=TEST_SIZE,random_state=RANDOM_STATE)
        logging.info(f'X_train: {X_train.shape},X_test: {X_test.shape},y_train: {y_train.shape},y_test: {y_test.shape}')
        #Initializing a machine learning model

        with mlflow.start_run(run_name="SVCC"):
           # Find best hyper parameters using a grid search
           param_grid = {'C': [0.1, 1],  
              'gamma': [1, 0.1, 0.01], 
              'kernel': ['rbf']}#, 'poly', 'sigmoid']}
           
           #svcc = GridSearchCV(SVC(), param_grid, cv = 5, refit = True, n_jobs=-1)
           svcc = SVC(gamma='auto')
           #cv = 5
           # Train the model on the whole train set
           svcc.fit(X_train,y_train)
  
           print("Searching for best parameters...")
           print(svcc.get_params)
           
           y_pred=svcc.predict(X_test)
           signature = infer_signature(X_train, y_pred)
           #crr = classification_report(y_test,y_pred)

           
           crr = classification_report(y_test, y_pred, output_dict=True)
           print(crr)
           crr = pd.DataFrame(crr).transpose()
           print(confusion_matrix(y_test,y_pred))
           accuracy = accuracy_score(y_test,y_pred)
           precision = precision_score(y_test, y_pred, average='weighted')
           recall = recall_score(y_test, y_pred, average='weighted')
           f1 = f1_score(y_test, y_pred, average='weighted')

           mlflow.log_metric('accuracy_score',accuracy)
           mlflow.log_metric('precision_score', precision)
           mlflow.log_metric('recall_score', recall)
           mlflow.log_metric('f1_score', f1)

           logging.info('Trained model evaluation done using validation data.')

           # Saving the calculated metrics into a json file in the Metrics folder
           metrics_folder = param['metrics_path']['metrics_folder']
           metrics_filename = param['metrics_path']['metrics_file']

           Utility().create_folder(metrics_folder)

           with open(os.path.join(metrics_folder, metrics_filename), 'w') as json_file:
                metrics = dict()
                metrics['accuracy_score'] = accuracy
                metrics['precision_score'] = precision
                metrics['recall_score'] = recall
                metrics['f1_score'] = f1

                json.dump(metrics, json_file, indent=4)

           clf_report_path = param['metrics_path']['clf_report_filename']

           crr.to_csv(os.path.join(metrics_folder, clf_report_path))

           logging.info('Saved evaluations in files.')

           # Saving the trained machine learing model in the models folder
           model_foldername = param['data_location']['main_model_folder']
           model_name = param['data_location']['model_name1']

           Utility().create_folder(model_foldername)
           model_path = os.path.join(model_foldername, model_name)

           joblib.dump(svcc, model_path)

           logging.info('Trained model saved as a joblib file.')

           # For Remote server (AWS)(DAGShub)
           #remort_server_uri = "https://dagshub.com/Hsinghsudwal/Mlflow-Dagshub.mlflow"
           #mlflow.set_tracking_uri(remort_server_uri)
           tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
           if tracking_url_type_store != "file":
              mlflow.sklearn.log_model(svcc, "model", registered_model_name="Grid Search cv(SVC)")
           else:
              mlflow.sklearn.log_model(svcc, "model")
         
      except Exception as e:
        logging.error(e)
        raise e


def main():
   model = Model()
   model.model_training()

if __name__ == "__main__":
    #testing
    main()