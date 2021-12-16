import numpy as np

class LinearRegression:
    '''
    A class which implements simple linear regression model.
    '''
    def __init__(self):
        self.b0 = None
        self.b1 = None
    
    def fit(self, X, y):
        '''
        Used to calculate slope and intercept coefficients.
        
        :param X: array, single feature
        :param y: array, true values
        :return: None
        '''
        numerator = np.sum((X - np.mean(X)) * (y - np.mean(y)))
        denominator = np.sum((X - np.mean(X)) ** 2)
        self.b1 = numerator / denominator
        self.b0 = np.mean(y) - self.b1 * np.mean(X)
        
    def predict(self, X):
        '''
        Makes predictions using the simple line equation.
        
        :param X: array, single feature
        :return: None
        '''
        if not self.b0 or not self.b1:
            raise Exception('Please call `SimpleLinearRegression.fit(X, y)` before making predictions.')
        return self.b0 + self.b1 * X


X = np.arange(start=1, stop=100)
y = np.random.normal(loc=X, scale=20)

X_train = X[:80]
y_train = y[:80]
X_test = X[80:]
y_test = y[80:]

model = LinearRegression()
model.fit(X_train, y_train)

print("Actual Value:",y_test)
print("Predicted Value:", model.predict(X_test))

# Output:
#Actual Value: [ 74.03066646  56.8634165   92.76459729  85.39525975 100.26254604
#  110.82481729  98.46153229 103.92901549  79.1009431  142.73690583
#  119.09533062 106.13586638  82.56381059  96.20764483  97.37956816
#   88.39007849  89.39548661  86.91936648  86.09059372]
# Predicted Value: [81.17214841 82.17438085 83.17661328 84.17884572 85.18107815 86.18331058
#  87.18554302 88.18777545 89.19000788 90.19224032 91.19447275 92.19670519
#  93.19893762 94.20117005 95.20340249 96.20563492 97.20786736 98.21009979
#  99.21233222]