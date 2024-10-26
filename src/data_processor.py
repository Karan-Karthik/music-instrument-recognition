import os
import numpy as np
import librosa
import pickle
import torch
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
    mean = 0.5
    std = 0.5
    return (tensor - mean) / std

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
        mfcc = librosa.feature.mfcc(y = audio, sr= sample_rate, n_mfcc=130)
        rmse = librosa.feature.rms(y=audio)
        spectral_centroid = librosa.feature.spectral_centroid(y= audio, sr=sample_rate)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y = audio, sr= sample_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(y = audio, sr = sample_rate, roll_percent=0.8)
        zcr = librosa.feature.zero_crossing_rate(y=audio)

        
        return {
            'spectrogram': spectrogram_db,
            'mfcc': mfcc,
            'rmse':rmse,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
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

        # Function to pad each tensor
        def pad_tensor(tensor, target_rows):
            current_rows = tensor.shape[0]
            if current_rows < target_rows:
                # Pad with zeros if shorter
                padding = torch.zeros((target_rows - current_rows, tensor.shape[1]), device=device)  # Zero padding on the specified device
                return torch.cat((tensor, padding), dim=0)  # Concatenate along the first dimension
            else:
                return tensor[:target_rows, :]  # Truncate if longer

        for audio_data_tuple in audio_data:
            spectrogram_dict = self.convert_to_spectrogram(audio_data_tuple)

            # Normalize and pad tensors
            # Pad each tensor to the target size
            spectrogram_tensor = pad_tensor(normalize(torch.tensor(spectrogram_dict['spectrogram'], dtype=torch.float32, device= device)), target_rows)
            mfcc_tensor = pad_tensor(normalize(torch.tensor(spectrogram_dict['mfcc'], dtype=torch.float32, device= device)), target_rows)
            rmse_tensor = pad_tensor(normalize(torch.tensor(spectrogram_dict['rmse'], dtype=torch.float32, device= device)), target_rows)
            spectral_centroid_tensor = pad_tensor(normalize(torch.tensor(spectrogram_dict['spectral_centroid'], dtype=torch.float32, device= device)), target_rows)
            spectral_bandwidth_tensor = pad_tensor(normalize(torch.tensor(spectrogram_dict['spectral_bandwidth'], dtype=torch.float32, device= device)), target_rows)
            spectral_rolloff_tensor = pad_tensor(normalize(torch.tensor(spectrogram_dict['spectral_rolloff'], dtype=torch.float32, device= device)), target_rows)
            zcr_tensor = pad_tensor(normalize(torch.tensor(spectrogram_dict['zcr'], dtype=torch.float32, device= device)), target_rows)

            preprocessed_data.append({
                'spectrogram': spectrogram_tensor,
                'mfcc': mfcc_tensor,
                'rmse': rmse_tensor,
                'spectral_centroid': spectral_centroid_tensor,
                'spectral_bandwidth': spectral_bandwidth_tensor,
                'spectral_rolloff': spectral_rolloff_tensor,
                'zcr': zcr_tensor,
                'instrument_tag': spectrogram_dict['instrument_tag']  # Save instrument tag
            })

        return preprocessed_data


    def preprocess_test_spectrograms(self, audio_data):
        preprocessed_data = []

        for audio_data_tuple in audio_data:
            
            spectrogram_tensor = normalize(torch.tensor(audio_data_tuple['spectrogram'], dtype=torch.float32))
            mfcc_tensor = normalize(torch.tensor(audio_data_tuple['mfcc'], dtype=torch.float32))
            rmse_tensor = normalize(torch.tensor(audio_data_tuple['rmse'], dtype=torch.float32))
            spectral_centroid_tensor = normalize(torch.tensor(audio_data_tuple['spectral_centroid'], dtype=torch.float32))
            spectral_bandwidth_tensor = normalize(torch.tensor(audio_data_tuple['spectral_bandwidth'], dtype=torch.float32))
            spectral_rolloff_tensor = normalize(torch.tensor(audio_data_tuple['spectral_rolloff'], dtype=torch.float32))
            zcr_tensor = normalize(torch.tensor(audio_data_tuple['zcr'], dtype=torch.float32))

            preprocessed_data.append({
                'spectrogram': spectrogram_tensor,
                'mfcc': mfcc_tensor,
                'rmse': rmse_tensor,
                'spectral_centroid': spectral_centroid_tensor,
                'spectral_bandwidth': spectral_bandwidth_tensor,
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
        self.save_preprocessed_data(processed_train_data, self.data_transform_config.processed_train_path)

        logging.info("Saving processed test data")
        self.save_preprocessed_data(processed_test_data, self.data_transform_config.processed_test_path)
        logging.info("Data transformation completed.")
