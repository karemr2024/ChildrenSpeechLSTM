# featureExtraction.py

import librosa
import numpy as np

def extract_features(file_path, max_pad_len=40):
    try:
        # Load the file with the specified sample rate of 48 kHz (48000 Hz)
        audio, sample_rate = librosa.load(file_path, sr=48000, mono=True, res_type='kaiser_fast')
        # Extract MFCCs with the same sample rate
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        # Pad or truncate the MFCCs to have a consistent shape
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            # Pad if the file is shorter than the expected length
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        elif pad_width < 0:
            # Truncate if the file is longer
            mfccs = mfccs[:, :max_pad_len]
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}, error: {e}")
        # Use a zero array to maintain consistency in case of an error
        mfccs = np.zeros((40, max_pad_len))
    return mfccs
