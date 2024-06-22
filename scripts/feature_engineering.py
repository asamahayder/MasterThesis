import numpy as np
import logger

def feature_engineer(data, params):
    final_tukey_data = data[0]
    labels = data[1]

    # Fitting polynomial to extract features
    # Degree of the polynomial
    degree = 15

    polynomial_coefficients = []

    
    for pulse in final_tukey_data:
        x = np.arange(len(pulse)) 
        y = pulse 
        
        
        coeffs = np.polyfit(x, y, degree)
        polynomial_coefficients.append(coeffs)

    
    polynomial_coefficients = np.array(polynomial_coefficients)

    logger.log("Shape of data after polynomial fit: ", polynomial_coefficients.shape)

    # printing unique labels and their counts
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        logger.log(f"{label}: {count}")


    X = polynomial_coefficients
    y = labels

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