from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

def feature_engineer(data, n_pcs):
    X = data[0]
    y = data[1]

    # PCA
    pca = PCA(n_components=None) # Keeping all components

    pca.fit(X)

    X_pca = pca.transform(X)

    X_pca_10 = X_pca[:, 0:n_pcs] # Keeping first n pcs

    # Simple split (No k-fold and validation set yet. Later do leave-one-out fold)
    # Creating test and training sets
    # Have to be multiples of 10 to ensure that all 10 pulses within a datapoint is in the same set
    # The reason is that 10 consequtive pulses from same file is likely to be more correlated compared to pulses from other files.
    # Therefore if we have pulses from same file in both test and training it can create unrealisticly good performance!
    # The term for this is data leakage

    label_encoder = LabelEncoder()
    label_encoder.fit(y)

    y_encoded = label_encoder.transform(y) # class labels as integers

    X_train = X_pca_10[:420]
    y_train = y_encoded[:420]

    X_test = X_pca_10[420:]
    y_test = y_encoded[420:]

    #The last 20 pulses have 10 pulses from PBS and 10 pulses from g/PBS
    #This is important so we can test both classes

    
    return X_train, y_train, X_test, y_test
