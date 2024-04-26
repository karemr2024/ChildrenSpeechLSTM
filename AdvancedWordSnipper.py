import numpy as np
import soundfile as sf
import whisper
import os

# Load your Whisper model
model = whisper.load_model("base")

def high_pass_filter(signal, samplerate, cutoff=100, order=5):
    from scipy.signal import butter, lfilter
    nyq = 0.5 * samplerate
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal

def detect_voice_activity(signal, samplerate, frame_length=0.02, threshold=0.02):
    frame_size = int(samplerate * frame_length)
    energy = np.array([np.sum(np.abs(signal[i:i+frame_size])**2) for i in range(0, len(signal), frame_size)])
    max_energy = np.max(energy)
    vad = energy > (max_energy * threshold)
    return vad

def get_word_segments(vad, samplerate, frame_length=0.02):
    frame_size = int(samplerate * frame_length)
    segments = []
    current_segment = None
    for i, active in enumerate(vad):
        if active and current_segment is None:
            current_segment = [i * frame_size]
        elif not active and current_segment is not None:
            current_segment.append(i * frame_size)
            segments.append(tuple(current_segment))
            current_segment = None
    if current_segment is not None:  # Catch any segment that goes till end
        current_segment.append(len(vad) * frame_size)
        segments.append(tuple(current_segment))
    return segments

def transcribe_audio(audio_path):
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    result = whisper.decode(model, mel)

    return result.text

def process_audio_files(input_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate over all WAV files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            audio_path = os.path.join(input_dir, filename)
            print(f"Processing file: {audio_path}")
            main(audio_path, output_dir)

def main(audio_path, output_dir):
    data, samplerate = sf.read(audio_path)
    filtered_signal = high_pass_filter(data, samplerate)
    vad = detect_voice_activity(filtered_signal, samplerate)
    word_segments = get_word_segments(vad, samplerate)
    transcribed_text = transcribe_audio(audio_path)
    words = transcribed_text.split()

    # Adjust if number of detected segments and words don't match
    segments_per_word = len(word_segments) // len(words)

    for i, word in enumerate(words):
        start_frame = word_segments[i * segments_per_word][0]
        end_frame = word_segments[(i + 1) * segments_per_word - 1][1]
        word_data = filtered_signal[start_frame:end_frame]
        output_path = os.path.join(output_dir, f"{word}_{i}.wav")
        sf.write(output_path, word_data, samplerate)
        print(f"Saved word segment: {output_path}")

input_directory = 'snipper_input'  # Path to the directory containing WAV files
output_directory = 'snipper_output'  # Path to the output directory
process_audio_files(input_directory, output_directory)
