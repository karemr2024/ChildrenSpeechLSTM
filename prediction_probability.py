import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from data_loader import load_data

# Load the model and test data
model = load_model('speech_recognition_model.keras')
_, X_test, _, y_test = load_data()

# Predict probabilities for the test set
probabilities = model.predict(X_test)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Overall test accuracy: {test_accuracy:.4f}")

# Get the true class labels (assuming y_test is not one-hot encoded)
true_labels = np.argmax(y_test, axis=1)

# Select a subset of samples randomly or based on some criterion
subset_indices = np.random.choice(len(X_test), size=4, replace=False)
subset_probabilities = probabilities[subset_indices]
subset_true_labels = true_labels[subset_indices]

# Plot the probability distributions for the subset of samples
fig, axes = plt.subplots(len(subset_indices), 1, figsize=(10, 20))
for i, ax in enumerate(axes):
    ax.bar(range(probabilities.shape[1]), subset_probabilities[i])
    true_label = subset_true_labels[i]
    ax.set_title(f'Sample {subset_indices[i]} (True Label: {true_label}) Probability Distribution')
    ax.set_xlabel('Class')
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1)

plt.tight_layout()
plt.show()

