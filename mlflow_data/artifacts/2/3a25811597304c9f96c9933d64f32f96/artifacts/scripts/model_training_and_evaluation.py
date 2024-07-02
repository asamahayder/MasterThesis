from sklearn.ensemble import RandomForestClassifier
import numpy as np
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import logging
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def train_and_evaluate_model(data, parameters):
    X, y = data

    params = parameters

    k_outer = params['k_outer']
    k_inner = params['k_inner']

    outer_cv = KFold(n_splits=k_outer, shuffle=True, random_state=42)

    training_accuracies = []
    test_accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    final_confusion_matrix = np.zeros((2, 2))
    true_labels = []
    predicted_labels = []
    params_list = []
    models = []

    # This is needed to convert back from 0s and 1s to the original labels
    le = LabelEncoder()
    le.classes_ = np.array(["g/PBS", "PBS"])

    # Printing unique labels and their counts
    logger.info("Class counts:")
    unique_labels, counts = np.unique(y, return_counts=True)
    unique_labels = le.inverse_transform(unique_labels)
    for label, count in zip(unique_labels, counts):
        logger.info(f"{label}: {count}")

    logger.info(f"Number of total datapoints: {len(y)}")

    def process_outer_fold(train_outer_indicies, test_outer_indicies, outer_fold_number):
        X_train_outer, X_test_outer = X[train_outer_indicies], X[test_outer_indicies]
        y_train_outer, y_test_outer = y[train_outer_indicies], y[test_outer_indicies]

        def objective(trial):
            validation_accuracies = []

            n_estimators = trial.suggest_int('n_estimators', params['n_estimators']['min'], params['n_estimators']['max'])
            max_depth = trial.suggest_int('max_depth', params['max_depth']['min'], params['max_depth']['max'])
            min_samples_split = trial.suggest_int('min_samples_split', params['min_samples_split']['min'], params['min_samples_split']['max'])
            min_samples_leaf = trial.suggest_int('min_samples_leaf', params['min_samples_leaf']['min'], params['min_samples_leaf']['max'])
            bootstrap = trial.suggest_categorical('bootstrap', params['bootstrap'])
            criterion = trial.suggest_categorical('criterion', params['criterion'])

            model_for_tuning = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                bootstrap=bootstrap,
                criterion=criterion,
                random_state=42,
            ) 

            inner_cv = KFold(n_splits=k_inner, shuffle=True, random_state=42)

            for train_inner_indicies, test_inner_indicies in inner_cv.split(X_train_outer, y_train_outer):
                X_train_inner, X_test_inner = X_train_outer[train_inner_indicies], X_train_outer[test_inner_indicies]
                y_train_inner, y_test_inner = y_train_outer[train_inner_indicies], y_train_outer[test_inner_indicies]

                model_for_tuning.fit(X_train_inner, y_train_inner)

                validation_accuracy = model_for_tuning.score(X_test_inner, y_test_inner)
                validation_accuracies.append(validation_accuracy)

            return np.mean(validation_accuracies)
        
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(), sampler=optuna.samplers.TPESampler(seed=42)) 
        study.optimize(objective, n_trials=params['optuna_n_trials'])

        best_params = study.best_params

        # Training the model on the outer fold using best parameters
        model_outer = RandomForestClassifier(random_state=42, **best_params)
        model_outer.fit(X_train_outer, y_train_outer)

        y_train_outer_pred = model_outer.predict(X_train_outer)
        y_test_outer_pred = model_outer.predict(X_test_outer)
        y_pred_test_outer_proba = model_outer.predict_proba(X_test_outer)[:, 1]

        training_accuracy = accuracy_score(y_train_outer, y_train_outer_pred)
        test_accuracy = accuracy_score(y_test_outer, y_test_outer_pred)

        precision = precision_score(y_test_outer, y_test_outer_pred)
        recall = recall_score(y_test_outer, y_test_outer_pred)
        cm = confusion_matrix(y_test_outer, y_test_outer_pred)
        f1 = f1_score(y_test_outer, y_test_outer_pred)
        roc_auc = roc_auc_score(y_test_outer, y_pred_test_outer_proba)

        # Logging
        logger.info(f"Outer fold number: {outer_fold_number}")

        # Logging the count of the true classes
        logger.info("Outer loop true class counts:")
        unique_labels, counts = np.unique(y_test_outer, return_counts=True)
        unique_labels = le.inverse_transform(unique_labels)
        for label, count in zip(unique_labels, counts):
            logger.info(f"{label}: {count}")

        # Logging the count of the prediction classes from training
        logger.info("Outer loop training predictions class counts:")
        unique_labels, counts = np.unique(y_train_outer_pred, return_counts=True)
        unique_labels = le.inverse_transform(unique_labels)
        for label, count in zip(unique_labels, counts):
            logger.info(f"{label}: {count}")
        
        # Logging the count of the prediction classes from testing
        logger.info("Outer loop testing predictions class counts:")
        unique_labels, counts = np.unique(y_test_outer_pred, return_counts=True)
        unique_labels = le.inverse_transform(unique_labels)
        for label, count in zip(unique_labels, counts):
            logger.info(f"{label}: {count}")

        # Logging the parameters
        logger.info("Best parameters:")
        logger.info(best_params)

        # Logging the training and test accuracies
        logger.info(f"Training accuracy: {training_accuracy}")
        logger.info(f"Test accuracy: {test_accuracy}")
        
        logger.info("")

        return (training_accuracy, test_accuracy, precision, recall, f1, cm, y_test_outer, y_test_outer_pred, best_params, model_outer)

    outer_results = Parallel(n_jobs=-1)(
        delayed(process_outer_fold)(train_outer_indicies, test_outer_indicies, outer_fold_number)
        for outer_fold_number, (train_outer_indicies, test_outer_indicies) in enumerate(tqdm(outer_cv.split(X, y), desc='Outer CV Split', total=k_outer), start=1)
    )

    for result in outer_results:
        training_accuracy, test_accuracy, precision, recall, f1, cm, y_test_outer, y_test_outer_pred, best_params, model_outer = result

        training_accuracies.append(training_accuracy)
        test_accuracies.append(test_accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        final_confusion_matrix += cm
        true_labels.extend(y_test_outer)
        predicted_labels.extend(y_test_outer_pred)
        params_list.append(best_params)
        models.append(model_outer)
        
    # Doing one final parameter search on the entire training set to get best possible model for production
    # However, we cannot use this final model to estimate the accuracy as it has seen the entire training set

    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', params['n_estimators']['min'], params['n_estimators']['max'])
        max_depth = trial.suggest_int('max_depth', params['max_depth']['min'], params['max_depth']['max'])
        min_samples_split = trial.suggest_int('min_samples_split', params['min_samples_split']['min'], params['min_samples_split']['max'])
        min_samples_leaf = trial.suggest_int('min_samples_leaf', params['min_samples_leaf']['min'], params['min_samples_leaf']['max'])
        bootstrap = trial.suggest_categorical('bootstrap', params['bootstrap'])
        criterion = trial.suggest_categorical('criterion', params['criterion'])

        model_for_tuning = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            bootstrap=bootstrap,
            criterion=criterion,
            random_state=42,
        ) 
        
        model_for_tuning.fit(X, y)
        score = model_for_tuning.score(X, y)

        return score
    
    final_study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(), sampler=optuna.samplers.TPESampler(seed=42)) 
    final_study.optimize(objective, n_trials=params['optuna_n_trials'], n_jobs=-1)

    final_hyperparameters = final_study.best_params

    model_final = RandomForestClassifier(random_state=42, **final_hyperparameters)
    model_final.fit(X, y)

    # Averaging the final metrics
    final_training_accuracy = np.mean(training_accuracies)
    final_test_accuracy = np.mean(test_accuracies)
    final_precision = np.mean(precisions)
    final_recall = np.mean(recalls)
    TP, FP, FN, TN = final_confusion_matrix.ravel()
    final_specificity = TN / (TN + FP)
    final_sensitivity = TP / (TP + FN)
    final_f1 = np.mean(f1_scores)

    cm_display = ConfusionMatrixDisplay(final_confusion_matrix, display_labels=le.classes_)
    cm_display.plot()
    plt.savefig("temp_plots/confusion_matrix.png")
    plt.close()

    # Plotting ROC
    fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig("temp_plots/roc_curve.png")
    plt.close()

    logger.info(f"Final estimated training accuracy by averaging: {final_training_accuracy}")
    logger.info(f"**** Final estimated test accuracy by averaging: {final_test_accuracy} ****")
    logger.info(f"Final estimated precision by averaging: {final_precision}")
    logger.info(f"Final estimated recall by averaging: {final_recall}")
    logger.info(f"Final estimated specificity by averaging: {final_specificity}")
    logger.info(f"Final estimated sensitivity by averaging: {final_sensitivity}")
    logger.info(f"Final estimated f1 score by averaging: {final_f1}")

    metrics = {}
    metrics['final_training_accuracy'] = final_training_accuracy
    metrics['final_test_accuracy'] = final_test_accuracy
    metrics['final_precision'] = final_precision
    metrics['final_recall'] = final_recall
    metrics['final_specificity'] = final_specificity
    metrics['final_sensitivity'] = final_sensitivity
    metrics['final_f1'] = final_f1

    return model_final, metrics