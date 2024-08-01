### Created a polynomial-regression for the relationship between the ZA fuel price between 2010 - 2015 ###


# Imports
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# Training set
X_test = [[2010], [2011], [2012], [2013], [2014], [2015], [2016]] # Year range
Y_test = [[7], [8], [9], [10], [11], [12], [13]] # Price range

# Testing set
X_train = [[2010], [2011], [2012], [2013], [2014], [2015]] # Year range
Y_train = [[7.67], [8.58], [10.43], [11.63], [13.36], [11.02]] # Price range


# Training the Linear Regression model and plotting a prediction
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
xx = np.linspace(2009, 2016, 100) # 2009 being the starting point, 2016 the ending point and 100 typically being equally spaced numbers between 2009 and 2016
yy = regressor.predict(xx.reshape(xx.shape[0], 1))  # Reshaping the input data into the correct format and then making predictions using the trained regression model
plt.plot(xx, yy)


# Setting the degree of the Polynomial Regression model
quadratic_featurizer = PolynomialFeatures(degree=2)


# This preprocessor transforms an input data matrix into a new data matrix of a given degree
X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
X_test_quadratic = quadratic_featurizer.transform(X_test)


# Training and test the regressor_quadratic model
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(X_train_quadratic, Y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))


# Plotting the graph, adding labels and showing
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c='r', linestyle='--')
plt.title('Petrol Price 2010 - 2020')
plt.xlabel('Years')
plt.ylabel('Price in Rands')
plt.axis([2010, 2016, 6, 14])
plt.grid(True)
plt.scatter(X_train, Y_train)
plt.show()


# This function had to be used due to my Y_train having decimal points. The numbers were rounded and 'infinite' output numbers removed
def print_rounded_array(arr):
    formatted_array = np.array2string(np.round(arr).astype(int), 
                                       formatter={'all': lambda x: f'{x}'}, 
                                       separator=', ')    
    formatted_array = formatted_array.replace('e+00', '').replace('e-00', '')
    print(formatted_array)


# Printing the outputs
print (X_train)
print_rounded_array(X_train_quadratic)
print (X_test)
print_rounded_array(X_test_quadratic)