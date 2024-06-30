from sklearn.ensemble import RandomForestClassifier
import numpy as np
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import logger
from tqdm import tqdm

def process_inner_fold(train_inner_indicies, test_inner_indicies, X_train_outer, y_train_outer, params):
    X_train_inner, X_test_inner = X_train_outer[train_inner_indicies], X_train_outer[test_inner_indicies]
    y_train_inner, y_test_inner = y_train_outer[train_inner_indicies], y_train_outer[test_inner_indicies]

    # Hyperparameter tuning using optuna and the reduced features
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', params['n_estimators']['min'], params['n_estimators']['max'])
        max_depth = trial.suggest_int('max_depth', params['max_depth']['min'], params['max_depth']['max'])
        min_samples_split = trial.suggest_int('min_samples_split', params['min_samples_split']['min'], params['min_samples_split']['max'])
        min_samples_leaf = trial.suggest_int('min_samples_leaf', params['min_samples_leaf']['min'], params['min_samples_leaf']['max'])
        max_features = trial.suggest_categorical('max_features', params['max_features'])
        bootstrap = trial.suggest_categorical('bootstrap', params['bootstrap'])
        criterion = trial.suggest_categorical('criterion', params['criterion'])
        min_weight_fraction_leaf = trial.suggest_float('min_weight_fraction_leaf', params['min_weight_fraction_leaf']['min'], params['min_weight_fraction_leaf']['max'])
        max_leaf_nodes = trial.suggest_int('max_leaf_nodes', params['max_leaf_nodes']['min'], params['max_leaf_nodes']['max'])
        min_impurity_decrease = trial.suggest_float('min_impurity_decrease', params['min_impurity_decrease']['min'], params['min_impurity_decrease']['max'])

        model_for_tuning = RandomForestClassifier(
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
            random_state=42,
        )

        model_for_tuning.fit(X_train_inner, y_train_inner)
        y_pred = model_for_tuning.predict(X_test_inner)
        accuracy = accuracy_score(y_test_inner, y_pred)
        return accuracy
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=params['optuna_n_trials'])

    best_params = study.best_params
    return best_params

def train_and_evaluate_model(data, parameters):
    X_train, y_train, X_test, y_test = data

    params = parameters

    k_outer = params['k_outer']
    k_inner = params['k_inner']

    outer_cv = KFold(n_splits=k_outer, shuffle=True, random_state=42)

    accuracies = []
    final_params = []

    for train_outer_indicies, test_outer_indicies in tqdm(outer_cv.split(X_train, y_train), desc='Outer CV Split', total=k_outer):
        X_train_outer, X_test_outer = X_train[train_outer_indicies], X_train[test_outer_indicies]
        logger.log(f"train_outer_indicies: {train_outer_indicies}")
        logger.log(f"test_outer_indicies: {test_outer_indicies}")
        y_train_outer, y_test_outer = y_train[train_outer_indicies], y_train[test_outer_indicies]

        inner_cv = KFold(n_splits=k_inner, shuffle=True, random_state=42)

        results = Parallel(n_jobs=-1)(delayed(process_inner_fold)(train_inner_indicies, test_inner_indicies, X_train_outer, y_train_outer, params) for train_inner_indicies, test_inner_indicies in tqdm(inner_cv.split(X_train_outer, y_train_outer), desc='Inner CV Split', total=k_inner, leave=False))

        # Combining results from parallel processing
        best_params_from_inner = [result for result in results]

        # Averaging the best hyperparameters from inner folds
        best_params = {}
        for key in best_params_from_inner[0].keys():
            if isinstance(best_params_from_inner[0][key], str) or isinstance(best_params_from_inner[0][key], bool):
                best_params[key] = max(set([params[key] for params in best_params_from_inner]), key=[params[key] for params in best_params_from_inner].count)
            elif isinstance(best_params_from_inner[0][key], int):
                best_params[key] = int(np.round(np.mean([params[key] for params in best_params_from_inner])))
            else:
                best_params[key] = np.mean([params[key] for params in best_params_from_inner])

        final_params.append(best_params)

        # Training the model on the outer fold
        model_outer = RandomForestClassifier(random_state=42, **best_params)
        model_outer.fit(X_train_outer, y_train_outer)
        y_pred_outer = model_outer.predict(X_test_outer)
        accuracy_outer = accuracy_score(y_test_outer, y_pred_outer)
        accuracies.append(accuracy_outer)

    # Averaging the final accuracies
    final_accuracy = np.mean(accuracies)
    logger.log(f"Final estimated accuracy by averaging: {final_accuracy}")

    # Averaging the final hyperparameters
    final_hyperparameters = {}
    for key in final_params[0].keys():
        if isinstance(final_params[0][key], str) or isinstance(final_params[0][key], bool):
            final_hyperparameters[key] = max(set([params[key] for params in final_params]), key=[params[key] for params in final_params].count)
        elif isinstance(final_params[0][key], int):
            final_hyperparameters[key] = int(np.round(np.mean([params[key] for params in final_params])))
        else:
            final_hyperparameters[key] = np.mean([params[key] for params in final_params])

    logger.log(f"Final hyperparameters: {final_hyperparameters}")

    # Training the final model on the entire training set
    model_final = RandomForestClassifier(random_state=42, **final_hyperparameters)
    model_final.fit(X_train, y_train)

    # Evaluating the final model on the test set
    y_pred_test = model_final.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    # Logging some random predictions and actual values

    logger.log("Random predictions and actual values:")
    random_indices = np.random.choice(len(y_test), 10, replace=False)
    for i in random_indices:
        logger.log(f"Prediction: {y_pred_test[i]}, Actual: {y_test[i]}")

    logger.log(f"Final accuracy on the test set: {accuracy_test}")

    return model_final, accuracy_test, final_hyperparameters
