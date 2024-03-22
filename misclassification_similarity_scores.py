import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from data_loader import load_data

# Load your model and test data
model = load_model('speech_recognition_model.keras')
_, X_test, _, y_test = load_data()

# Predict probabilities for each class on test data
probabilities = model.predict(X_test)

# Get the predicted classes based on the highest probability
predicted_classes = np.argmax(probabilities, axis=1)

# Get the true classes from the test labels
true_classes = np.argmax(y_test, axis=1)

# Extract the maximum probability for each sample
max_probabilities = np.max(probabilities, axis=1)

# Identify the indices of the samples that were misclassified
misclassified_indices = np.where(predicted_classes != true_classes)[0]

# First, get the max probabilities for misclassified samples
misclassified_max_probs = max_probabilities[misclassified_indices]

threshold = 0.95  # This value could be adjusted based on your distribution analysis

# Then, apply the threshold condition to these probabilities
misclassification_similarity_scores = misclassified_max_probs[misclassified_max_probs <= threshold]

# Optional: Perform further analysis on misclassification_similarity_scores
# For example, you could calculate the average similarity score for misclassifications
average_similarity_score_for_misclassifications = np.mean(misclassification_similarity_scores)

# Identify misclassified probabilities that are above the threshold
above_threshold = misclassified_max_probs > threshold

# Calculate the percentage of misclassifications above the threshold
percentage_above_threshold = np.mean(above_threshold) * 100

print(f"Percentage of misclassifications above the threshold ({threshold}): {percentage_above_threshold:.2f}%")

print(f"Average similarity score for misclassifications: {average_similarity_score_for_misclassifications:.4f}")

# You can also visualize the distribution of the maximum probabilities
import matplotlib.pyplot as plt

plt.hist(misclassified_max_probs, bins=50, alpha=0.75)
plt.axvline(threshold, color='red', linestyle='dashed', linewidth=1)
plt.title('Distribution of Maximum Prediction Probabilities for Misclassifications')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.show()
