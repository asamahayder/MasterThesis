import numpy as np
from numpy.polynomial import Polynomial
import logger
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

""" def fit_polynomial(pulse, degree):
    x = np.arange(len(pulse))
    y = pulse
    coeffs = np.polyfit(x, y, degree)
    return coeffs """

def fit_polynomial(pulse, degree):
    x = np.arange(len(pulse))
    y = pulse
    # Fit the polynomial
    p = Polynomial.fit(x, y, degree)
    # Convert to the standard basis and get the coefficients
    coeffs = p.convert().coef
    return coeffs

def feature_engineer(data, params):
    final_tukey_data = data[0]
    labels = data[1]
    reserved_data = data[2]
    reserved_labels = data[3]

    # Fitting polynomial to extract features
    # Degree of the polynomial
    number_of_pcs = params["number_of_pcs"]

    # Parallel fitting polynomial to extract features
    """ polynomial_coefficients = Parallel(n_jobs=-1)(
        delayed(fit_polynomial)(pulse, degree) for pulse in tqdm(final_tukey_data, desc="Fitting polynomial to extract features")
    )

    polynomial_coefficients_reserved = Parallel(n_jobs=-1)(
        delayed(fit_polynomial)(pulse, degree) for pulse in tqdm(reserved_data, desc="Fitting polynomial to extract features (reserved data)")
    )

    
    polynomial_coefficients = np.array(polynomial_coefficients)
    polynomial_coefficients_reserved = np.array(polynomial_coefficients_reserved)
 """
    



    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # This step will standardize the data
        ('pca', PCA(n_components=None))  # This step will apply PCA
    ])  # Create the pipeline object

    pipeline.fit(final_tukey_data)

    X_pca = pipeline.transform(final_tukey_data)

    X_pca_reserved = pipeline.transform(reserved_data)




    # Plotting Cumulative Explained Variance

    pca = pipeline.named_steps['pca']

    explained_variance = pca.explained_variance_ratio_

    cumulitive_explained_variance = 0
    most_important_components = []

    required_variance_explained = .95

    for i, component in enumerate(explained_variance):
        if cumulitive_explained_variance > required_variance_explained:
            break
        cumulitive_explained_variance += component
        most_important_components.append(pca.components_[i])


    # Plotting the explained variance
    plt.figure(figsize=(8, 5))
    plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid', label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig("temp_plots/PCA_cumulative_variance_explained.png")

    logger.log(f"Number of components needed to explain {required_variance_explained*100}% of variance is {len(most_important_components)}")
    logger.log(f"First 2 components explain {(explained_variance[0] + explained_variance[1])*100}% of variance")



    # Plotting data onto first 2 PCs

    

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    # Plotting data onto first 2 PCs for different classes
    unique_classes = np.unique(labels)
    for cls in unique_classes:
        ix = [i for i in range(len(y)) if y[i] == cls]
        ax.scatter(X_pca[ix, 0], X_pca[ix, 1], label=cls, s=50)

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('PCA of Dataset labeled by class')
    ax.legend()

    plt.tight_layout()
    plt.show()



















    X = X_pca[:, 0:number_of_pcs]
    y = labels

    logger.log("Shape of data after PCA fit: ", X_pca.shape)
    logger.log("")

    # printing unique labels and their counts
    logger.log("Class counts in training set:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        logger.log(f"{label}: {count}")

    logger.log(f"Number of total datapoints: {len(y) + len(reserved_labels)}")

    # 4 datapoints has already been reserverd for final evaluation
    X_train = X
    y_train = y
    X_test = X_pca_reserved[:, 0:number_of_pcs]
    y_test = reserved_labels

    # printing the number of datapoints in the training and testing sets
    logger.log(f"Number of training datapoints: {len(y_train)}")
    logger.log(f"Number of testing datapoints: {len(y_test)}")

    # logging each distinct class in y_test along with their counts
    logger.log("Class counts in test set:")
    unique_labels, counts = np.unique(y_test, return_counts=True)
    for label, count in zip(unique_labels, counts):
        logger.log(f"{label}: {count}")
    logger.log("")

    
    return X_train, y_train, X_test, y_test