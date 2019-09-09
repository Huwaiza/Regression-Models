# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#encoding the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEncoder_X  = LabelEncoder()
X[:,3] = labelEncoder_X.fit_transform(X[:,3]) 

#creating dummies of independent variables
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#avoiding dummy variable trap
#to avoid redundant data but the python libraries are taking care of it but sometimes you have to manually remove it
X = X[:, 1:] 

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#multiple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting the test results
y_pred = regressor.predict(X_test)

#Building the optimal model using the backward elimination method
import statsmodels.formula.api as sm
#adding 1 column of all ones  since  y = b0 + bx0 + .. + bnxn
X = np.append(arr = np.ones((50,1)).astype(int) , values  = X , axis = 1 )

X_opt = X[: , [0,1,2,3,4,5]]
#step 2 of backward elimination
regressor_ols = sm.OLS(endog = y , exog = X_opt).fit()
regressor_ols.summary()

X_opt = X[: , [0,1,3,4,5]]
regressor_ols = sm.OLS(endog = y , exog = X_opt).fit()
regressor_ols.summary()

X_opt = X[: , [0,3,4,5]]
regressor_ols = sm.OLS(endog = y , exog = X_opt).fit()
regressor_ols.summary()

X_opt = X[: , [0,3,5]]
regressor_ols = sm.OLS(endog = y , exog = X_opt).fit()
regressor_ols.summary()

X_opt = X[: , [0,3]]
regressor_ols = sm.OLS(endog = y , exog = X_opt).fit()
regressor_ols.summary()




