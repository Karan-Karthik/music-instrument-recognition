import os
import numpy as np
import librosa
import pickle
from dataclasses import dataclass
from logger import logging
from data_processor import DataTransformation


#from exceptions.DataException import k

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
        train_dir = '/Users/karanwork/Documents/Deep learning project/music-instrument-recognition/data/train'
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
        Load all instrument test data by pairing each audio file with its corresponding .txt file,
        then save the spectrograms and associated metadata.
        """
        test_dir = '/Users/karanwork/Documents/Deep learning project/music-instrument-recognition/data/test'
        logging.info('Entered test data ingestion method')

        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Test directory not found: {test_dir}")

        audio_files = []
        text_data = []

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
                        
                        # Find corresponding .txt file
                        txt_filename = file.replace('.wav', '.txt')
                        txt_filepath = os.path.join(part_dir, txt_filename)
                        if os.path.exists(txt_filepath):
                            # Load the instrument tag from the .txt file
                            text = self.load_txt_file(txt_filepath).strip()
                            text_data.append(f"File: {file}, Text: {text}")
                            
                            # Convert the audio file to a spectrogram
                            spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=130)
                            spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
                            
                            # Add spectrogram, sample rate, and instrument tag to the audio_files list
                            audio_files.append({
                                'spectrogram': spectrogram_db,
                                'filename': file,
                                'instrument_tag': text  # Save the instrument tag from .txt file
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

    #modeltrainer= ModelTrainer()