import numpy as np
import logger

def feature_engineer(data, params):
    X = data[0]
    y = data[1]

    # Fitting polynomial to extract features
    # Degree of the polynomial
    degree = params['degree_of_polynomial']

    # To store polynomial coefficients for each pulse
    polynomial_coefficients = []

    # Fit polynomial for each pulse
    for pulse in X:
        x = np.arange(len(pulse))  # x-axis values (0, 1, 2, ..., 2999)
        y = pulse  # y-axis values are the pulse data
        
        # Fit polynomial to the data
        coeffs = np.polyfit(x, y, degree)
        polynomial_coefficients.append(coeffs)

    # Convert list to numpy array for easier manipulation
    polynomial_coefficients = np.array(polynomial_coefficients)

    logger.log("Shape of data after polynomial fit: ", polynomial_coefficients.shape)

    # print unique labels and their counts
    unique_labels, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique_labels, counts):
        logger.log(f"{label}: {count}")


    X = polynomial_coefficients


    logger.log(f"Number of total datapoints: {len(y)}")

    # Reserving 4 datapoints for the final evaluation
    X_train = X[:-4]
    y_train = y[:-4]
    X_test = X[-4:]
    y_test = y[-4:]

    # printing the number of datapoints in the training and testing sets
    logger.log(f"Number of training datapoints: {len(y_train)}")
    logger.log(f"Number of testing datapoints: {len(y_test)}")

    # checking that the reserved data points are from different samples and equal classes
    logger.log(f"Classes in the reserved datapoints: {y_test}")

    
    return X_train, y_train, X_test, y_test