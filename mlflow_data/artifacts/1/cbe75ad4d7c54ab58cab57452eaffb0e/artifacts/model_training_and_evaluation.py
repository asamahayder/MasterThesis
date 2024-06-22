from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score

def train_and_evaluate_model(data, params):

    X_train = data[0]
    y_train = data[1]
    X_test = data[2]
    y_test = data[3]

    knn_to_test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 20, 32, 64]

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

    print(f"Test Accuracy: {test_accuracy} with knn {knn_for_max_accuracy}")

    return model, test_accuracy
