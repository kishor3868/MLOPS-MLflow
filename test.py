# testing the url of the file
import mlflow 
print(mlflow.get_tracking_uri()) #output : file:///F:/MLOps/MLOPS-MLflow/mlruns
# thus we need to change this output from file to https
mlflow.set_tracking_uri('http://localhost:5000')
print(mlflow.get_tracking_uri())
# thus changes to https