import os
import numpy as np
import librosa
import pickle
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torchvision import transforms
from logger import logging
from dataclasses import dataclass

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

"""
Create classes for all these tasks then call it in data_loader
call it so when i call data_loader it does everything below (__init__)
get data from artifacts which should have been there from data_loader
convert to spectrograms
Run other transforms on it
save to pickle file
then start model trainer
"""

def normalize(tensor):
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / (std + 1e-6)  # Adding a small value to avoid division by zero

# Function to add a small amount of noise
def add_noise(data, noise_level=0.001):
    noise = noise_level * np.random.normal(size=data.shape)
    return data + noise


@dataclass
class DataTransformationConfig:
    processed_train_path: str = os.path.join('artifacts', 'processed_train_audio_files.pkl')
    processed_test_path: str = os.path.join('artifacts', 'processed_test_audio_files.pkl')
    raw_train_path: str = os.path.join('artifacts', 'raw_train_audio_files.pkl')
    raw_test_path: str = os.path.join('artifacts', 'raw_test_audio_files.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transform_config = DataTransformationConfig()
        # Define the PyTorch transformation pipeline
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts to tensor
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalizes the spectrogram
        ])


    def convert_to_spectrogram(self, audio_data_tuple):

        audio, sample_rate, instrument_tag = audio_data_tuple

        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=130)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        # Normalize spectrogram_db
        spectrogram_db_normalized = spectrogram_db / np.max(np.abs(spectrogram_db))
        spectrogram_db = add_noise(spectrogram_db_normalized)
                                

        mfcc = librosa.feature.mfcc(y = audio, sr= sample_rate, n_mfcc=130)
        # Normalize MFCC
        scaler = StandardScaler()
        mfcc_normalized = scaler.fit_transform(mfcc.T).T
        mfcc = add_noise(mfcc_normalized)

        rmse = librosa.feature.rms(y=audio)
        # Normalize RMSE
        rmse_normalized = (rmse - np.mean(rmse)) / np.std(rmse)
        rmse = add_noise(rmse_normalized)

        spectral_centroid = librosa.feature.spectral_centroid(y= audio, sr=sample_rate)
        spectral_centroid = add_noise(spectral_centroid)

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y = audio, sr= sample_rate)
        spectral_bandwidth = add_noise(spectral_bandwidth)

        spectral_contrast = librosa.feature.spectral_contrast(S=spectrogram, sr=sample_rate)
        # Normalize spectral contrast
        scaler = MinMaxScaler()
        spectral_contrast_normalized = scaler.fit_transform(spectral_contrast)
        spectral_contrast = add_noise(spectral_contrast_normalized)

        spectral_rolloff = librosa.feature.spectral_rolloff(y = audio, sr = sample_rate, roll_percent=0.8)
        spectral_rolloff = add_noise(spectral_rolloff)

        zcr = librosa.feature.zero_crossing_rate(y=audio)
        # Normalize ZCR
        zcr_normalized = (zcr - np.min(zcr)) / (np.max(zcr) - np.min(zcr))
        zcr = add_noise(zcr_normalized)

        
        return {
            'spectrogram': spectrogram_db,
            'mfcc': mfcc,
            'rmse':rmse,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_contrast': spectral_contrast,
            'spectral_rolloff': spectral_rolloff,
            'zcr': zcr,
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


    def preprocess_training_spectrograms(self, audio_data):
        preprocessed_data = []
        
        # Define the target size for padding
        target_rows = 130  # Adjust this based on your requirements


        for audio_data_tuple in audio_data:

            spectrogram_dict = self.convert_to_spectrogram(audio_data_tuple)
            # Normalize and pad tensors
            # Pad each tensor to the target size
            spectrogram_tensor = spectrogram_dict['spectrogram']
            mfcc_tensor = spectrogram_dict['mfcc']
            rmse_tensor = spectrogram_dict['rmse']
            spectral_centroid_tensor = spectrogram_dict['spectral_centroid']
            spectral_bandwidth_tensor = spectrogram_dict['spectral_bandwidth']
            spectral_contrast_tensor = spectrogram_dict['spectral_contrast']
            spectral_rolloff_tensor = spectrogram_dict['spectral_rolloff']
            zcr_tensor = spectrogram_dict['zcr']


            preprocessed_data.append({
                'spectrogram': spectrogram_tensor,
                'mfcc': mfcc_tensor,
                'rmse': rmse_tensor,
                'spectral_centroid': spectral_centroid_tensor,
                'spectral_bandwidth': spectral_bandwidth_tensor,
                'spectral_contrast': spectral_contrast_tensor,
                'spectral_rolloff': spectral_rolloff_tensor,
                'zcr': zcr_tensor,
                'instrument_tag': spectrogram_dict['instrument_tag']  # Save instrument tag
            })

        return preprocessed_data


    def preprocess_test_spectrograms(self, audio_data):
        preprocessed_data = []

        for audio_data_tuple in audio_data:
            
            spectrogram_tensor = normalize(torch.tensor(np.array(audio_data_tuple['spectrogram']), dtype=torch.float32))
            mfcc_tensor = normalize(torch.tensor(np.array(audio_data_tuple['mfcc']), dtype=torch.float32))
            rmse_tensor = normalize(torch.tensor(np.array(audio_data_tuple['rmse']), dtype=torch.float32))
            spectral_centroid_tensor = normalize(torch.tensor(np.array(audio_data_tuple['spectral_centroid']), dtype=torch.float32))
            spectral_bandwidth_tensor = normalize(torch.tensor(np.array(audio_data_tuple['spectral_bandwidth']), dtype=torch.float32))
            spectral_contrast_tensor = normalize(torch.tensor(np.array(audio_data_tuple['spectral_contrast']), dtype=torch.float32))
            spectral_rolloff_tensor = normalize(torch.tensor(np.array(audio_data_tuple['spectral_rolloff']), dtype=torch.float32))
            zcr_tensor = normalize(torch.tensor(np.array(audio_data_tuple['zcr']), dtype=torch.float32))

            preprocessed_data.append({
                'spectrogram': spectrogram_tensor,
                'mfcc': mfcc_tensor,
                'rmse': rmse_tensor,
                'spectral_centroid': spectral_centroid_tensor,
                'spectral_bandwidth': spectral_bandwidth_tensor,
                'spectral_contrast': spectral_contrast_tensor,
                'spectral_rolloff': spectral_rolloff_tensor,
                'zcr': zcr_tensor,
                'filename': audio_data_tuple['filename']
            })
        return preprocessed_data


    def save_preprocessed_data(self, preprocessed_data, save_path):
        """
        Save preprocessed data to pickle file.
        """
        with open(save_path, 'wb') as f:
            pickle.dump(preprocessed_data, f)
        logging.info(f"Processed data saved to {save_path}")


    def transforms(self):
        """
        This method downloads the raw pickle files, generates spectrograms, applies preprocessing
        transformations, and saves the processed data into new pickle files in the artifacts folder.
        """
        # Step 1: Load the raw train and test pickle files
        logging.info("Loading raw train and test data to data processor")
        raw_train_data = self.load_pickle_file(self.data_transform_config.raw_train_path)
        raw_test_data = self.load_pickle_file(self.data_transform_config.raw_test_path)

        # Step 2: Convert raw data to spectrograms and apply preprocessing
        logging.info("Processing training data")
        processed_train_data = self.preprocess_training_spectrograms(raw_train_data)

        logging.info("Processing testing data")
        processed_test_data = self.preprocess_test_spectrograms(raw_test_data)

        
        # Step 3: Save the processed data to pickle files
        logging.info("Saving processed train data")
        #self.save_preprocessed_data(processed_train_data, self.data_transform_config.processed_train_path)

        self.save_preprocessed_data(processed_train_data, 'artifacts/processed_train_audio_files.pkl')

        logging.info("Saving processed test data")
        self.save_preprocessed_data(processed_test_data, self.data_transform_config.processed_test_path)
        logging.info("Data transformation completed.")
