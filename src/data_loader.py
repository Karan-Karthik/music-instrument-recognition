import os
import numpy as np
import librosa
import pickle
from dataclasses import dataclass
from logger import logging
from data_processor import DataTransformation
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#from exceptions.DataException import k
# Function to add a small amount of noise
def add_noise(data, noise_level=0.001):
    noise = noise_level * np.random.normal(size=data.shape)
    return data + noise


@dataclass
class DataIngestionConfig:
    raw_train_path: str = os.path.join('artifacts', 'raw_train_audio_files.pkl')
    raw_test_path: str = os.path.join('artifacts', 'raw_test_audio_files.pkl')
    raw_test_txt_path: str = os.path.join('artifacts', 'raw_test_audio_files.txt')  # Path for the text file


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()


    def load_audio(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Audio file not found: {filepath}")
    
        audio, sample_rate = librosa.load(filepath, sr= 44100)
        return audio, sample_rate


    def load_txt_file(self, filepath):

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Text file not found: {filepath}")

        with open(filepath, 'r') as f:
            text = f.readline().strip()
        return text


    def load_audio_file(self, path, filename):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Train data directory not found: {path}")

        if filename.endswith('.wav'):  # Check if it's a .wav file
            filepath = os.path.join(path, filename)
            # Load audio file using librosa
            audio, sample_rate = self.load_audio(filepath)
            return audio, sample_rate


    def load_instruments_train_data(self):
        train_dir = 'data/train'
        logging.info('Entered training data ingestion method')

        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Train directory not found: {train_dir}")

        audio_files=[]

        for instrument_folder in os.listdir(train_dir):
            instrument_dir = os.path.join(train_dir, instrument_folder)

            if os.path.isdir(instrument_dir):
                instrument_tag = instrument_folder

                for file in os.listdir(instrument_dir):
                    if file.endswith(".wav"):
                        # Load audio using self.load_audio_file
                        audio, sample_rate = self.load_audio_file(instrument_dir, file)
                        audio_files.append((audio, sample_rate, instrument_tag))

        # Save the list using pickle
        with open(self.ingestion_config.raw_train_path, 'wb') as f:
            pickle.dump(audio_files, f)

        logging.info("Loaded train Data")


    def load_instruments_test_data(self):
        """
        Load all instrument test data by splitting audio files into 0.2-second chunks if they are longer than 0.2 seconds,
        then save the spectrograms, MFCC, and RMSE features, and associated metadata.
        """
        test_dir = 'data/test'
        logging.info('Entered test data ingestion method')

        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Test directory not found: {test_dir}")

        audio_files = []
        text_data = []
        chunk_duration = 3  # duration of each chunk in seconds

        # Loop through each Part (e.g., Part1, Part2, Part3)
        for part_folder in os.listdir(test_dir):
            part_dir = os.path.join(test_dir, part_folder)

            if os.path.isdir(part_dir):
                # Iterate through each file in the part folder
                for file in os.listdir(part_dir):
                    if file.endswith(".wav"):
                        # Load audio file
                        audio_filepath = os.path.join(part_dir, file)
                        audio, sample_rate = self.load_audio(audio_filepath)
                        audio_duration = librosa.get_duration(y=audio, sr=sample_rate)
                        
                        # Find corresponding .txt file
                        txt_filename = file.replace('.wav', '.txt')
                        txt_filepath = os.path.join(part_dir, txt_filename)
                        if os.path.exists(txt_filepath):
                            # Load the instrument tag from the .txt file
                            text = self.load_txt_file(txt_filepath).strip()
                            text_data.append(f"File: {file}, Text: {text}")

                            # Split audio into 0.2-second chunks if it's longer than 0.2 seconds
                            num_chunks = int(np.ceil(audio_duration / chunk_duration))
                            for i in range(num_chunks):
                                
                                start_sample = int(i * chunk_duration * sample_rate)
                                end_sample = int(min((i + 1) * chunk_duration * sample_rate, len(audio)))
                                audio_chunk = audio[start_sample:end_sample]

                                # Skip this chunk if it's less than 0.2 seconds
                                if (end_sample - start_sample) / sample_rate < 2.8:
                                    continue
                                
                                # Convert the audio chunk to a spectrogram
                                # Visual representation of the frequency content of an audio signal over time
                                spectrogram = librosa.feature.melspectrogram(y=audio_chunk, sr=sample_rate, n_mels=130)
                                spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
                                
                                # Normalize spectrogram_db
                                spectrogram_db_normalized = spectrogram_db / np.max(np.abs(spectrogram_db))
                                spectrogram_db = add_noise(spectrogram_db_normalized)
                                
                                
                                # Compute MFCC and RMSE for the chunk
                                # MFCC: Coefficients that represent the short-term power spectrum of an audio signal on a non-linear Mel scale of frequency
                                # RMSE: energy or loudness of an audio signal in each frame or segment
                                mfcc = librosa.feature.mfcc(y=audio_chunk, sr=sample_rate, n_mfcc=130)

                                # Normalize MFCC
                                scaler = StandardScaler()
                                mfcc_normalized = scaler.fit_transform(mfcc.T).T
                                mfcc = add_noise(mfcc_normalized)
                                

                                rmse = librosa.feature.rms(y=audio_chunk)
                                # Normalize RMSE
                                rmse_normalized = (rmse - np.mean(rmse)) / np.std(rmse)
                                rmse = add_noise(rmse_normalized)

                                # Compute spectral centroid
                                # Indicates the center of mass of the spectrum, often associated with the "brightness" of a sound
                                spectral_centroid = librosa.feature.spectral_centroid(y=audio_chunk, sr=sample_rate)
                                # Normalize spectral centroid
                                spectral_centroid = add_noise(spectral_centroid)

                                # Compute spectral bandwidth
                                # Measures the range of frequencies in a sound, helpful in distinguishing instruments with broader vs. narrower frequency ranges
                                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_chunk, sr=sample_rate)
                                # Normalize spectral bandwidth
                                spectral_bandwidth = add_noise(spectral_bandwidth)


                                # Compute spectral contrast
                                # Captures the difference in amplitude between peaks and valleys in the spectrum, useful for identifying texture differences
                                spectral_contrast = librosa.feature.spectral_contrast(S=spectrogram, sr=sample_rate)
                                # Normalize spectral contrast
                                scaler = MinMaxScaler()
                                spectral_contrast_normalized = scaler.fit_transform(spectral_contrast)
                                spectral_contrast = add_noise(spectral_contrast_normalized)

                                # Compute spectral rolloff (e.g., rolloff at 85% of energy)
                                # The frequency below which a certain percentage (e.g., 85%) of the spectral energy is contained, differentiating high-pitched vs. low-pitched sounds
                                spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_chunk, sr=sample_rate, roll_percent=0.8)
                                spectral_rolloff = add_noise(spectral_rolloff)

                                # Compute zero-crossing rate (ZCR)
                                # Counts the number of times the signal crosses zero amplitude, often higher for percussive sounds
                                zcr = librosa.feature.zero_crossing_rate(y=audio_chunk)
                                # Normalize ZCR
                                zcr_normalized = (zcr - np.min(zcr)) / (np.max(zcr) - np.min(zcr))
                                zcr = add_noise(zcr_normalized)

                                # Add chunk data to the audio_files list
                                audio_files.append({
                                    'spectrogram': spectrogram_db,
                                    'mfcc': mfcc,
                                    'rmse': rmse,
                                    'spectral_centroid': spectral_centroid,
                                    'spectral_bandwidth': spectral_bandwidth,
                                    'spectral_contrast': spectral_contrast,
                                    'spectral_rolloff': spectral_rolloff,
                                    'zcr': zcr,
                                    'filename': f"{file}_chunk_{i}"  # Keep the original file name with chunk index
                                })
                        else:
                            logging.warning(f"Text file not found for {file}")

        # Save the processed audio files (spectrograms and metadata) using pickle
        with open(self.ingestion_config.raw_test_path, 'wb') as f:
            pickle.dump(audio_files, f)

        # Save the text data (for logging or debugging purposes) to a .txt file
        with open(self.ingestion_config.raw_test_txt_path, 'w') as f:
            for line in text_data:
                f.write(line + "\n")
        
        logging.info("Test data loaded and saved")


if __name__== '__main__':
    # Create an object of DataIngestion class
    data_ingestion = DataIngestion()
    # Load and save the training audio files
    data_ingestion.load_instruments_train_data()
    # Load and save the test audio files and their corresponding text files
    data_ingestion.load_instruments_test_data()
    # Create an object of DataTransformation class
    data_transformation = DataTransformation()
    # Perform transformation, generate spectrograms, and save to pickle
    data_transformation.transforms()
