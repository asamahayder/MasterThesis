import importlib
import json
import mlflow
import os
import atexit
import shutil

mlflow.set_tracking_uri("http://localhost:5000")
current_file_directory = os.path.dirname(os.path.abspath(__file__))
temp_folder_path = os.path.join(current_file_directory, 'temp_plots')
os.mkdir(temp_folder_path)

# This ensures that the temp folder will be deleted regardless if there is an error or not
def cleanup(current_file_directory):
    mlflow.log_artifacts(os.path.join(current_file_directory, 'temp_plots'), artifact_path="plots")
    mlflow.log_artifact(os.path.join(current_file_directory, 'log.txt'))
    
    mlflow.end_run()

    if os.path.exists(temp_folder_path):
        shutil.rmtree(temp_folder_path)
    if os.path.exists("log.txt"):
        os.remove("log.txt")

atexit.register(cleanup, current_file_directory)

# Setting the current working directory to the directory of the run script
os.chdir(current_file_directory)

# Load configuration
with open('config.json') as f:
    config = json.load(f)

def run_step(module_name, function_name, data, params):
    print("Running", module_name, function_name)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    return function(data, params)

# Start MLflow run
mlflow.start_run(
    experiment_id=1, 
    run_name="Reintroduced alignment with PCA 50 pcs", 
    tags={"algorithm": "Random Forest", "domain": "Time Domain"}, 
    description="Reintroducing alignment now with PCA. I expect the number of needed PCs to be less than 50 now that the alignment reduces variance."
)

mlflow.log_artifact(os.path.join(current_file_directory, 'config.json'))
mlflow.log_artifact(os.path.join(current_file_directory, 'run.py'), artifact_path="scripts")
mlflow.log_artifact(os.path.join(current_file_directory, 'data_load.py'), artifact_path="scripts")
mlflow.log_artifact(os.path.join(current_file_directory, 'preprocessing.py'), artifact_path="scripts")
mlflow.log_artifact(os.path.join(current_file_directory, 'feature_engineering.py'), artifact_path="scripts")
mlflow.log_artifact(os.path.join(current_file_directory, 'model_training_and_evaluation.py'), artifact_path="scripts")

data_load_params = config['data_load']['params']
for param, value in data_load_params.items():
    mlflow.log_param(param, value)

preprocess_params = config['preprocessing']['params']
for param, value in preprocess_params.items():
    mlflow.log_param(param, value)

feature_engineering_params = config['feature_engineering']['params']
for param, value in feature_engineering_params.items():
    mlflow.log_param(param, value)

model_training_params = config['model_training_and_evaluation']['params']
for param, value in model_training_params.items():
    mlflow.log_param(param, value)


# Step 1: Load data
data = run_step('data_load', 'load_data', None, config['data_load']['params'])

# Step 2: Preprocess data
X, y, reserved_data, reserved_labels = run_step('preprocessing', 'preprocess_data', data, config['preprocessing']['params'])

# Step 3: Feature engineering
X_train, y_train, X_test, y_test = run_step('feature_engineering', 'feature_engineer', (X, y, reserved_data, reserved_labels), config['feature_engineering']['params'])

# Step 4: Model training and evaluation
model, accuracy, best_params = run_step('model_training_and_evaluation', 'train_and_evaluate_model', (X_train, y_train, X_test, y_test), config['model_training_and_evaluation']['params'])

mlflow.log_metric("accuracy", accuracy)
mlflow.sklearn.log_model(model, "model")
