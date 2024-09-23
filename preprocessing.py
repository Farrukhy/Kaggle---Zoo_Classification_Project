#This file contains all the data loading, splitting, and scaling steps, written without functions.

# rc/preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Loading the dataset
zoo_data = pd.read_csv("/kaggle/input/zoo-animal-classification/zoo.csv")

# Separating features (X) and labels (y)
X = zoo_data.iloc[:, :-1].values  # all columns except the last one
y = zoo_data.iloc[:, -1].values   # the last column (class label)

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scaling the features (dropping the first column which is 'animal_name')
X_train_scaled = X_train[:, 1:]
X_test_scaled = X_test[:, 1:]

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train_scaled)
X_test_scaled = sc.transform(X_test_scaled)
