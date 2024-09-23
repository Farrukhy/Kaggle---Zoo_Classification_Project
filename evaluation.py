#This file contains the evaluation part of the code, including the confusion matrix and cross-validation scores.

# rc/evaluation.py

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt

# Predicting the test set results
y_pred = classification.predict(X_test_scaled)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Classification report
print(classification_report(y_test, y_pred))

# Cross-validation
accuracies = cross_val_score(estimator=classification, X=X_train_scaled, y=y_train, cv=10)
print(f"Cross-validation mean accuracy: {accuracies.mean()}")
print(f"Cross-validation std deviation: {accuracies.std()}")

# Visualizing the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=[1, 2, 4, 5, 6, 7], yticklabels=[1, 2, 4, 5, 6, 7])
plt.title('Confusion Matrix for Zoo Classification')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
