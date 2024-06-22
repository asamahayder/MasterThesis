from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score
import optuna

def objective(trial):
    # Define the hyperparameters to tune
    n_neighbors = trial.suggest_int('n_neighbors', params["n_neighbors"]["min"], params["n_neighbors"]["max"])
    weights = trial.suggest_categorical('weights', params["weights"])
    algorithm = trial.suggest_categorical('algorithm', params["algorithm"])
    leaf_size = trial.suggest_int('leaf_size', params["leaf_size"]["min"], params["leaf_size"]["max"])
    p = trial.suggest_int('p', params["p"]["min"], params["p"]["max"])
    metric = trial.suggest_categorical('metric', params["metric"])

    # Create the model
    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        leaf_size=leaf_size,
        p=p,
        metric=metric
    )

    # This is needed to ensure that 10 consequitve pulses from same sample end up together
    groups = np.repeat(np.arange(X_train.shape[0] // 10), 10)
    gkf = GroupKFold(n_splits=10)

    # Evaluate the model using cross-validation
    accuracy = cross_val_score(knn, X_train, y_train, groups=groups, cv=gkf, scoring='accuracy').mean()

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
    print("Best score: ", study.best_value)

    best_model = KNeighborsClassifier(**study.best_params)
    best_model.fit(X_train, y_train)

    test_accuracy = best_model.score(X_test, y_test)

    return best_model, test_accuracy, study.best_params





"""knn_to_test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 20, 32, 64]

    accuracies = []
    models = []

    groups = np.repeat(np.arange(X_train.shape[0] // 10), 10)
    gkf = GroupKFold(n_splits=10)

    for k in knn_to_test:
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        cv_scores = cross_val_score(knn_classifier, X_train, y_train, groups=groups, cv=gkf, scoring='accuracy')
        accuracies.append(np.mean(cv_scores))
        models.append(knn_classifier)

    max_accuracy = max(accuracies)
    index_of_max = accuracies.index(max_accuracy)
    knn_for_max_accuracy = knn_to_test[index_of_max]
    model = models[index_of_max]

    # Now that we have the best model from the cross validation, we need to fit it to entire training data:
    model = model.fit(X_train, y_train)

    # Then we do the final evaluation on the test data. (So far only training data has been used)
    test_accuracy = model.score(X_test, y_test)

    print(f"Test Accuracy: {test_accuracy} with knn {knn_for_max_accuracy}")"""