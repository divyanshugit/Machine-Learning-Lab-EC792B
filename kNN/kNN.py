import numpy as np
from math import sqrt

class KNN():
    """ K Nearest Neighbors classifier.
    Parameters:
    -----------
    k: int
        The number of closest neighbors that will determine the class of the 
        sample that we wish to predict.
    """
    def __init__(self, k=5):
        self.k = k

    def euclidean_distance(self, x1, x2):
        """
        Calculate the euclidean distance between two rows.
        """ 
        distance = 0.0
        
        for i in range(len(x1)-1):
            distance += (x1[i] - x2[i])**2
            
        return sqrt(distance)
    def _vote(self, neighbor_labels):
        """ Return the most common class among the neighbor samples """
        counts = np.bincount(neighbor_labels.astype('int'))
        return counts.argmax()

    def predict(self, X_test, X_train, y_train):
        y_pred = np.empty(X_test.shape[0])
        # Determine the class of each sample
        for i, test_sample in enumerate(X_test):
            # Sort the training samples by their distance to the test sample and get the K nearest
            idx = np.argsort([self.euclidean_distance(test_sample, x) for x in X_train])[:self.k]
            # Extract the labels of the K nearest neighboring training samples
            k_nearest_neighbors = np.array([y_train[i] for i in idx])
            # Label sample as the most common class label
            y_pred[i] = self._vote(k_nearest_neighbors)

        return y_pred