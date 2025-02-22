# again we are doing all the things on the remote server on dagshub
# first we need to pip install dagshub then proceeds
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
dagshub.init(repo_owner='kishor3868', repo_name='MLOPS-MLflow', mlflow=True)

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_depth = 10
n_estimators = 12
 # all attributres of the mlflow are mlflow tracking api
# Mention your experiment below
# mlflow.set_experiment('YT-MLOPS-Exp1')
# changing the url from file to https otherwise you will get the error

mlflow.set_tracking_uri("https://dagshub.com/kishor3868/MLOPS-MLflow.mlflow ")

# You can log them in other experiment if it is already created like this
mlflow.set_experiment('EXP2') # if it is not created then code will create this
with mlflow.start_run():
    rf=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,random_state=42)
    rf.fit(X_train,y_train)
    
    y_pred=rf.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)

    # logging the required things
    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('max_depth',max_depth)
    mlflow.log_param('n_estimatiors',n_estimators)


    # Creating a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # save plot
    plt.savefig("Confusion-matrix.png")

    # # log artifacts using mlflow
    # for calling the artifacts we need to set the https from file
    mlflow.log_artifact('Confusion-matrix.png')
    mlflow.log_artifact(__file__)# loging the code as the file(script)


    # setting tag
    mlflow.set_tags({"Author": 'kishor', "Project": "Wine Classification"})

    # # Log the model
    mlflow.sklearn.log_model(rf, "Random-Forest-Model") 
    

    print(accuracy)