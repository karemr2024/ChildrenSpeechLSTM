import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def extract_features(file_path, max_pad_len=40):
    try:
        # Load the file with the specified sample rate of 48 kHz (48000 Hz)
        audio, sample_rate = librosa.load(file_path, sr=48000, mono=True, res_type='kaiser_fast')
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        # Pad or truncate the MFCCs
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        elif pad_width < 0:
            mfccs = mfccs[:, :max_pad_len]
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}, error: {e}")
        mfccs = np.zeros((40, max_pad_len))
    return audio, sample_rate, mfccs

def plot_waveform(audio, sample_rate, file_name='raw_audio_waveform.png'):
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, len(audio) / sample_rate, num=len(audio)), audio)
    plt.title('Raw Audio Waveform')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()
    
def plot_spectrogram(audio, sample_rate, file_name='spectrogram.png'):
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

def plot_mfccs(mfccs, sample_rate, file_name='mfcc.png'):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.xlabel('Time (seconds)')
    plt.ylabel('MFCC coefficients')
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

def main():
    file_path = 'more_7.wav'
    audio, sample_rate, mfccs = extract_features(file_path)
    plot_waveform(audio, sample_rate, 'raw_audio_waveform.png')
    plot_spectrogram(audio, sample_rate, 'spectrogram.png')
    plot_mfccs(mfccs, sample_rate, 'mfcc.png')

if __name__ == "__main__":
    main()
