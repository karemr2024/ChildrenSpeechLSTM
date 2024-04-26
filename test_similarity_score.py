import numpy as np
from tensorflow import keras
from sklearn.metrics import accuracy_score
from data_loader import load_data

# Load your model and test data
model = keras.models.load_model('speech_recognition_model.keras')
_, X_test, _, y_test = load_data()

# Predict probabilities for the test set
probabilities = model.predict(X_test)

# Calculate the maximum prediction probability for each sample
max_probabilities = np.max(probabilities, axis=1)

# Get the predicted class with the highest probability
predicted_classes = np.argmax(probabilities, axis=1)

# Get the true class labels (assuming y_test is one-hot encoded)
true_labels = np.argmax(y_test, axis=1)

# Calculate overall accuracy
accuracy = accuracy_score(true_labels, predicted_classes)
print(f"Overall test accuracy: {accuracy:.4f}")

# Define a threshold for prediction confidence
threshold = 0.95  # This value could be adjusted based on your distribution analysis

# Calculate the similarity scores for correctly classified samples
similarity_scores = max_probabilities[(predicted_classes == true_labels) & (max_probabilities >= threshold)]

# Calculate the average similarity score for correctly classified samples above the threshold
average_similarity_score = np.mean(similarity_scores)
print(f"Average similarity score for confident predictions: {average_similarity_score:.4f}")

# Calculate the percentage of similarity scores above the threshold
percentage_above_threshold = (similarity_scores.size / max_probabilities.size) * 100
print(f"Percentage of similarity scores above threshold: {percentage_above_threshold:.2f}%")


# You can also visualize the distribution of the maximum probabilities
import matplotlib.pyplot as plt

plt.hist(max_probabilities, bins=50, alpha=0.75)
plt.axvline(threshold, color='red', linestyle='dashed', linewidth=1)
plt.title('Distribution of Maximum Prediction Probabilities')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.show()
