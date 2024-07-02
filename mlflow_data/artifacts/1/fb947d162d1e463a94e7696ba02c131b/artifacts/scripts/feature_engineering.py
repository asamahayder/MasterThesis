import numpy as np
from numpy.polynomial import Polynomial
import logger
from tqdm import tqdm
from joblib import Parallel, delayed

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
    ptp = data[2]
    reserved_data = data[3]
    reserved_labels = data[4]
    reserved_ptp = data[5]

    # Fitting polynomial to extract features
    # Degree of the polynomial
    degree = params["degree_of_polynomial"]

    # Parallel fitting polynomial to extract features
    polynomial_coefficients = Parallel(n_jobs=-1)(
        delayed(fit_polynomial)(pulse, degree) for pulse in tqdm(final_tukey_data, desc="Fitting polynomial to extract features")
    )

    polynomial_coefficients_reserved = Parallel(n_jobs=-1)(
        delayed(fit_polynomial)(pulse, degree) for pulse in tqdm(reserved_data, desc="Fitting polynomial to extract features (reserved data)")
    )

    
    polynomial_coefficients = np.array(polynomial_coefficients)
    polynomial_coefficients_reserved = np.array(polynomial_coefficients_reserved)

    # adding shifts as a feature
    polynomial_coefficients = np.hstack((polynomial_coefficients, np.array(ptp).reshape(-1, 1)))
    polynomial_coefficients_reserved = np.hstack((polynomial_coefficients_reserved, np.array(reserved_ptp).reshape(-1, 1)))


    X = polynomial_coefficients
    y = labels

    logger.log("Shape of data after polynomial fit: ", polynomial_coefficients.shape)
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
    X_test = polynomial_coefficients_reserved
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