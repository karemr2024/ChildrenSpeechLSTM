import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import seaborn as sns
from data_loader import load_data

# Load the model and test data
model = load_model('speech_recognition_model.keras', compile=False)
_, X_test, _, y_test = load_data()

# Predict the probabilities for the test set and get the predicted class (with the highest probability)
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# If y_test is already one-hot encoded, get the true labels from it
true_labels = np.argmax(y_test, axis=1)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Calculate the test loss and test accuracy
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {test_loss:.4f}")
print(f"Overall test accuracy: {test_accuracy:.4f}")

# Visualize the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
