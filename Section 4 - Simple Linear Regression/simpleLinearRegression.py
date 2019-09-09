#Simple linear regression

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#fiiting simple linear regression model to the training model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train,1/3)

#predicting thr test set results
y_pred = regressor.predict(X_test)

#visualising the training result data
plt.scatter(X_train, y_train , color = 'red' )
plt.plot(X_train , regressor.predict(X_train) , color = 'blue' )
plt.title('Salary Vs Experience [Training set Result]')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#visualising the test result data
plt.scatter(X_test, y_test , color = 'red' )
plt.plot(X_train , regressor.predict(X_train) , color = 'blue' )
plt.title('Salary Vs Experience [Test set Result]')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()









