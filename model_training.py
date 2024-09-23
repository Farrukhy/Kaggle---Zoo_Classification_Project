#This file will train  LogisticRegression model without using functions.


# rc/model_training.py

from sklearn.linear_model import LogisticRegression

# Training the logistic regression model
classification = LogisticRegression(random_state=0)
classification.fit(X_train_scaled, y_train)
