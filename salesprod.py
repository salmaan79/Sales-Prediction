# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 11:11:55 2020

@author: Salmaan Ahmed Ansari
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 10:58:28 2020

@author: Salmaan Ahmed Ansari
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv', sep = ',')
X = dataset.iloc[:, 2:11].values
y = dataset.iloc[:, 11:12].values




# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(X[:, [0,1,3,4,5]])
X[:, [0,1,3,4,5]] = imputer.transform(X[:, [0,1,3,4,5]])

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer.fit(X[:, [2,6,7,8]])
X[:, [2,6,7,8]] = imputer.transform(X[:, [2,6,7,8]])


"""#taking care of missing data
dataset['Property_Area'].fillna(dataset['Property_Area'].mode()[0], inplace=True)
dataset['LoanAmount'].fillna(dataset['LoanAmount'].median(), inplace=True)
"""

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
le_X = LabelEncoder()
X[:, 0] = le_X.fit_transform(X[:, 0])
#doing onehotencoding for every multicategory variable and applying one dummy variable removal
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
X = X[:, 1:]
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [8])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
X = X[:, 1:]
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [10])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
X = X[:, 1:]







# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0)
regressor.fit(X, y)

"""# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = regressor,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)
"""





dataset_test = pd.read_csv('test.csv')




X_tes = dataset_test.iloc[:, 2:11].values




# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(X_tes[:, [0,1,3,4,5]])
X_tes[:, [0,1,3,4,5]] = imputer.transform(X_tes[:, [0,1,3,4,5]])

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer.fit(X_tes[:, [2,6,7,8]])
X_tes[:, [2,6,7,8]] = imputer.transform(X_tes[:, [2,6,7,8]])


"""#taking care of missing data
dataset['Property_Area'].fillna(dataset['Property_Area'].mode()[0], inplace=True)
dataset['LoanAmount'].fillna(dataset['LoanAmount'].median(), inplace=True)
"""

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
le_X = LabelEncoder()
X_tes[:, 0] = le_X.fit_transform(X_tes[:, 0])
#doing onehotencoding for every multicategory variable and applying one dummy variable removal
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X_tes = np.array(ct.fit_transform(X_tes))
X_tes = X_tes[:, 1:]
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [8])], remainder='passthrough')
X_tes = np.array(ct.fit_transform(X_tes))
X_tes = X_tes[:, 1:]
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [10])], remainder='passthrough')
X_tes = np.array(ct.fit_transform(X_tes))
X_tes = X_tes[:, 1:]






# Predicting the Test set results
y_tes = regressor.predict(X_tes)

y_tes



