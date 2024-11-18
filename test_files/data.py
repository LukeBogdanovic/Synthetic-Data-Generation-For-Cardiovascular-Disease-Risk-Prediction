from keras.api.models import load_model
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt

ecg_test_data = np.random.randn(100, 5000, 1)
crf_test_data = np.random.randn(100, 10)

# Load the pre-trained model
model_path = "test.keras"
model = load_model(model_path)

# Assuming ecg_test_data and crf_test_data are already defined
predictions = model.predict([ecg_test_data, crf_test_data])

# Print the raw prediction probabilities
print("Predicted probabilities:", predictions)

# Convert the predicted probabilities to binary class (0 or 1) using a threshold of 0.5
predicted_classes = (predictions > 0.5).astype(int)

# Print the predicted classes
print("Predicted classes (0 or 1):", predicted_classes)

# Generate true labels for the test data (for demonstration purposes)
true_labels = np.random.randint(2, size=(100,))

auc_score = roc_auc_score(true_labels, predictions)
print("Area under the curve (AUC): ", auc_score)

# Compute the confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_classes)

# Plot the confusion matrix
plt.figure(1,figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Heart Disease'], yticklabels=['Normal', 'Heart Disease'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
# plt.show()

fpr, tpr, _ = roc_curve(true_labels, predictions)
plt.figure(2,figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', label=f'AUC = {auc_score:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()