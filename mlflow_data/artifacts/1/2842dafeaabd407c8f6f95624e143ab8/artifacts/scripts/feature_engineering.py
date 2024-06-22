from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def feature_engineer(data, params):
    print("Running Feature Engineering")
    X = data[0]
    y = data[1]

    # PCA
    pca = PCA(n_components=None) # Keeping all components

    pca.fit(X)



    # Plotting Cumulative Explained Variance

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
    plt.savefig("temp_plots/pca_cumulative_variance.png")

    print(f"Number of components needed to explain {required_variance_explained*100}% of variance is {len(most_important_components)}")
    print(f"First 2 components explain {(explained_variance[0] + explained_variance[1])*100}% of variance")



    # Plotting data onto first 2 PCs
    X_pca = pca.transform(X)

    # Create the plot
    plt.figure(figsize=(12, 5))

    # Plotting data onto first 2 PCs for different classes
    unique_classes = np.unique(y)
    for cls in unique_classes:
        ix = [i for i in range(len(y)) if y[i] == cls]
        plt.scatter(X_pca[ix, 0], X_pca[ix, 1], label=cls, s=50)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Dataset labeled by class')
    plt.legend()

    plt.tight_layout()
    plt.savefig("temp_plots/pca_data.png")









    X_pca = pca.transform(X)

    X_pca_10 = X_pca[:, 0:params["n_pcs"]] # Keeping first n pcs

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