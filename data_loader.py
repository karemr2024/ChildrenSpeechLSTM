import numpy as np

def load_data():
     # Load the features and labels saved by dataPrep.py
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy').astype(int)  # Convert to integer type
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy').astype(int)  # Convert to integer type
    
    # One-hot encode the labels
    num_classes = len(np.unique(y_train))  # Assuming y_train includes all classes
    y_train_encoded = np.eye(num_classes)[y_train]
    y_test_encoded = np.eye(num_classes)[y_test]

    return X_train, X_test, y_train_encoded, y_test_encoded