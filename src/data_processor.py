import os
import numpy as np
import librosa
import pickle
from torchvision import transforms
from logger import logging
from dataclasses import dataclass



def normalize(tensor):
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / (std + 1e-6)  # Adding a small value to avoid division by zero


@dataclass
class DataTransformationConfig:
    processed_train_path: str = os.path.join('artifacts', 'processed_train_audio_files.pkl')
    raw_train_path: str = os.path.join('artifacts', 'raw_train_audio_files.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transform_config = DataTransformationConfig()

        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts to tensor
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
        ])

    def convert_to_mfccs(self, audio_data_tuple):
        audio, sample_rate, instrument_tag = audio_data_tuple

        # Perform amplitude normalization
        audio = audio / (np.max(np.abs(audio)) + 1e-10)

        # Perform silence removal
        audio, index = librosa.effects.trim(audio, top_db=20, frame_length= int(44100/250), hop_length=int(44100/500))

        # Step 2: Generate Spectrogram (Short-Time Fourier Transform)
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        spectrogram = np.abs(stft)

        # Step 3: Normalize the spectrogram
        spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
        spectrogram = (spectrogram - np.mean(spectrogram)) / (np.std(spectrogram) + 1e-6)  # Standard normalization

        # Generate MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=35)
        mfcc = np.mean(mfcc[1:, :], axis=1)  # Exclude the first coefficient
        mfcc = np.reshape(mfcc, (1, -1))

        return {
            'spectrogram': spectrogram,
            'mfcc': mfcc,
            'instrument_tag': instrument_tag
        }


    def load_pickle_file(self, filepath):
        """
        Load the pickle file from the given path.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Pickle file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        logging.info(f"Loaded data from {filepath}")
        return data


    def save_preprocessed_data(self, preprocessed_data, save_path):
        """
        Save preprocessed data to pickle file.
        """
        with open(save_path, 'wb') as f:
            pickle.dump(preprocessed_data, f)
        logging.info(f"Processed data saved to {save_path}")


    def transforms(self):

        logging.info("Loading raw data to data processor")
        raw_train_data = self.load_pickle_file(self.data_transform_config.raw_train_path)

        logging.info("Processing training data")
        processed_train_data = []
        for data_tuple in raw_train_data:
            processed_data = self.convert_to_mfccs(data_tuple)
            processed_train_data.append(processed_data)

        self.save_preprocessed_data(processed_train_data, self.data_transform_config.processed_train_path)
        logging.info("Processed train data saved successfully")
