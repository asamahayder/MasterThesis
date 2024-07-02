import numpy as np
from numpy.polynomial import Polynomial
import logger
from tqdm import tqdm
from joblib import Parallel, delayed

def fit_polynomial(pulse, degree):
    x = np.arange(len(pulse))
    y = pulse
    p = Polynomial.fit(x, y, degree)
    coeffs = p.convert().coef
    return coeffs

def feature_engineer(data, params):
    final_tukey_data, labels = data

    degree = params["degree_of_polynomial"]

    # Parallel fitting polynomial to extract features
    polynomial_coefficients = Parallel(n_jobs=-1)(
        delayed(fit_polynomial)(pulse, degree) for pulse in tqdm(final_tukey_data, desc="Fitting polynomial to extract features")
    )
    
    polynomial_coefficients = np.array(polynomial_coefficients)

    X = polynomial_coefficients
    y = labels

    logger.log("Shape of data after polynomial fit: ", polynomial_coefficients.shape)
    logger.log("")
    
    return X, y