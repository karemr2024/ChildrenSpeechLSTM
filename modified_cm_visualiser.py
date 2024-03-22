import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D

# Assuming data_loader.load_data() is a function that returns train and test data
from data_loader import load_data

# Load the model and test data
model = load_model('speech_recognition_model.keras')
_, X_test, _, y_test = load_data()

# Predict probabilities for the test set
probabilities = model.predict(X_test)

# Get the predicted class labels with the highest probability
predicted_labels = np.argmax(probabilities, axis=1)

# Get the true class labels (assuming y_test is one-hot encoded)
true_labels = np.argmax(y_test, axis=1)

# Calculate the overall accuracy
test_accuracy = np.mean(predicted_labels == true_labels)
print(f"Overall test accuracy: {test_accuracy:.4f}")

# Create the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Normalize the confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Obtain a flattened array of the prediction probabilities for the true class
prob_true_class = probabilities[np.arange(len(probabilities)), true_labels]

# Generate a figure with a 3D confusion matrix
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Define the grid for the x and y axis
x, y = np.meshgrid(range(cm.shape[1]), range(cm.shape[0]))

# Flatten the grid matrices
x = x.flatten()
y = y.flatten()
z = np.zeros(cm.size)

# Repeat each element of the normalized confusion matrix `prob_true_class.size` times
# and reshape it back to the confusion matrix size to have consistent z-axis values
dz = np.repeat(cm_normalized.flatten(), prob_true_class.size).reshape(cm.shape[0], cm.shape[1], -1)

# Plot the bars
for i in range(prob_true_class.size):
    ax.bar3d(x, y, z, 1, 1, dz[:,:,i].flatten(), shade=True)

ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
ax.set_zlabel('Normalized Count')

# Set the tick marks for x and y axis
ax.set_xticks(np.arange(0.5, cm.shape[1]))
ax.set_yticks(np.arange(0.5, cm.shape[0]))

# Set the tick labels for x and y axis
ax.set_xticklabels(range(cm.shape[1]))
ax.set_yticklabels(range(cm.shape[0]))

# Set the title for the plot
ax.set_title('Confusion Matrix with Prediction Probabilities')

plt.show()
