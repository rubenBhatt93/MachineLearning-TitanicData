# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset1 = pd.read_csv('train.csv')
X_train = dataset1.iloc[:, [2,4,5]].values
y_train = dataset1.iloc[:, 1].values
dataset2 = pd.read_csv('test.csv')
X_test = dataset2.iloc[:, [1,3,4]].values

# Taking care of missing data in X_train
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_train[:, 2:3])
X_train[:, 2:3] = imputer.transform(X_train[:, 2:3])

# Taking care of missing data in X_test
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_test[:, 2:3])
X_test[:, 2:3] = imputer.transform(X_test[:, 2:3])

# Encoding categorical data
# Encoding the Independent Variable of train data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_train[:, 1] = labelencoder_X.fit_transform(X_train[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X_train = onehotencoder.fit_transform(X_train).toarray()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_train[:, 2] = labelencoder_X.fit_transform(X_train[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [2])
X_train = onehotencoder.fit_transform(X_train).toarray()

#Taking n-1 dummy variables from each category in X_train
X_train = X_train[:, [0,1,3,5]]

# Encoding the Independent Variable of test data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_test = LabelEncoder()
X_test[:, 1] = labelencoder_X_test.fit_transform(X_test[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X_test = onehotencoder.fit_transform(X_test).toarray()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_test = LabelEncoder()
X_test[:, 2] = labelencoder_X_test.fit_transform(X_test[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [2])
X_test = onehotencoder.fit_transform(X_test).toarray()

#Taking n-1 dummy variables from each category in X_train
X_test = X_test[:, [0,1,3,5]]

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = np.array(classifier.predict(X_test))
y_pred = np.reshape(y_pred,(418,-1))

PredictedSurvival = dataset2.iloc[:, 0:1].values
PredictedSurvival = np.column_stack((PredictedSurvival,y_pred))