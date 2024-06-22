from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score
import optuna

def objective(trial):
    # Define the hyperparameters to tune
    n_estimators = trial.suggest_int('n_estimators', params["n_estimators"]["min"], params["n_estimators"]["max"])
    max_depth = trial.suggest_int('max_depth', params["max_depth"]["min"], params["max_depth"]["max"])
    min_samples_split = trial.suggest_int('min_samples_split', params["min_samples_split"]["min"], params["min_samples_split"]["max"])
    min_samples_leaf = trial.suggest_int('min_samples_leaf', params["min_samples_leaf"]["min"], params["min_samples_leaf"]["max"])
    max_features = trial.suggest_categorical('max_features', params["max_features"])
    bootstrap = trial.suggest_categorical('bootstrap', params["bootstrap"])
    criterion = trial.suggest_categorical('criterion', params["criterion"])
    min_weight_fraction_leaf = trial.suggest_float('min_weight_fraction_leaf', params["min_weight_fraction_leaf"]["min"], params["min_weight_fraction_leaf"]["max"])
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', params["max_leaf_nodes"]["min"], params["max_leaf_nodes"]["max"])
    min_impurity_decrease = trial.suggest_float('min_impurity_decrease', params["min_impurity_decrease"]["min"], params["min_impurity_decrease"]["max"])
    class_weight = trial.suggest_categorical('class_weight', params["class_weight"])

    # Create the model
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        criterion=criterion,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        class_weight=class_weight,
        random_state=42
    )

    # This is needed to ensure that 10 consecutive pulses from the same sample end up together
    groups = np.repeat(np.arange(X_train.shape[0] // 10), 10)
    gkf = GroupKFold(n_splits=10)

    # Evaluate the model using cross-validation
    accuracy = cross_val_score(rf, X_train, y_train, groups=groups, cv=gkf, scoring='accuracy').mean()

    return accuracy

def train_and_evaluate_model(data, parameters):
    global X_train, y_train, X_test, y_test, params

    print("Running Train and Evaluate Model")

    X_train, y_train, X_test, y_test = data

    params = parameters

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    # Print the best hyperparameters
    print("Best hyperparameters: ", study.best_params)
    print("Best accuracy from trials: ", study.best_value)

    best_model = RandomForestClassifier(**study.best_params)
    best_model.fit(X_train, y_train)

    test_accuracy = best_model.score(X_test, y_test)
    print("Best accuracy from test set: ", test_accuracy)

    return best_model, test_accuracy, study.best_params