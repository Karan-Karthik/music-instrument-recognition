import os
import numpy as np
import librosa
import pickle
import torch
from torchvision import transforms
from logger import logging
from dataclasses import dataclass

"""
Create classes for all these tasks then call it in data_loader
call it so when i call data_loader it does everything below (__init__)
get data from artifacts which should have been there from data_loader
convert to spectrograms
Run other transforms on it
save to pickle file
then start model trainer
"""


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
        return {
            'spectrogram': spectrogram_db,
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
        for audio_data_tuple in audio_data:
            spectrogram_dict = self.convert_to_spectrogram(audio_data_tuple)
            spectrogram_tensor = self.transform(spectrogram_dict['spectrogram'])  # Normalize and convert to tensor
            preprocessed_data.append({
                'spectrogram': spectrogram_tensor,
                'instrument_tag': spectrogram_dict['instrument_tag']  # Save instrument tag
            })
        return preprocessed_data
    
    def preprocess_test_spectrograms(self, audio_data):
        preprocessed_data = []
        for audio_data_tuple in audio_data:
            # Assuming audio_data_tuple contains 'spectrogram' and 'filename' (from the ingestion process)
            
            spectrogram_tensor = self.transform(audio_data_tuple['spectrogram'])  # Normalize and convert to tensor
            preprocessed_data.append({
                'spectrogram': spectrogram_tensor,
                'instrument_tag': audio_data_tuple['instrument_tag'],  # Save instrument tag for evaluation
                'filename': audio_data_tuple['filename']  # Save filename for matching
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
