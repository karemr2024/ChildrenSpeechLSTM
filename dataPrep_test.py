# dataPrep_test.py

import os
import numpy as np
from sklearn.model_selection import train_test_split
from featureExtraction import extract_features

def get_data_from_folder(folder_path, max_pad_len=40):
    features = []
    labels = []
    
    # Walk through the person directories
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.wav'):
                # Extract the label from the first character of the filename
                label = int(file[0])
                file_path = os.path.join(root, file)
                mfccs = extract_features(file_path, max_pad_len=max_pad_len)
                
                if mfccs is not None:
                    features.append(mfccs)
                    labels.append(label)
                    
    return np.array(features), np.array(labels)

def save_data(X, y, folder_name):
    # Save the features and labels into the specified folder
    np.save(os.path.join(folder_name, 'X.npy'), X)
    np.save(os.path.join(folder_name, 'y.npy'), y)

def main():
    data_folder_path = '/Users/emre/Desktop/ChildrenSpeechLSTM/data'
    train_data_path = '/Users/emre/Desktop/ChildrenSpeechLSTM/trainData'
    test_data_path = '/Users/emre/Desktop/ChildrenSpeechLSTM/testData'
    
    # Get features and labels from the data folder
    features, labels = get_data_from_folder(data_folder_path)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Create the trainData and testData folders if they don't exist
    os.makedirs(train_data_path, exist_ok=True)
    os.makedirs(test_data_path, exist_ok=True)
    
    # Save the split data
    save_data(X_train, y_train, train_data_path)
    save_data(X_test, y_test, test_data_path)

if __name__ == '__main__':
    main()
