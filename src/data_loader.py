import os
import librosa
import pickle
from dataclasses import dataclass
from logger import logging
from data_processor import DataTransformation




@dataclass
class DataIngestionConfig:
    raw_train_path: str = os.path.join('artifacts', 'raw_train_audio_files.pkl')



class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()


    def load_audio(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Audio file not found: {filepath}")
    
        audio, sample_rate = librosa.load(filepath, sr= 44100)
        return audio, sample_rate

    def load_audio_file(self, path, filename):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Train data directory not found: {path}")

        if filename.endswith('.mp3'):
            filepath = os.path.join(path, filename)

            audio, sample_rate = self.load_audio(filepath)
            return audio, sample_rate


    def load_instruments_train_data(self):
        train_dir = '/Users/karanwork/Documents/Deep learning project/music-instrument-recognition/data/london_phill_dataset_multi'
        logging.info('Entered training data ingestion method')

        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Train directory not found: {train_dir}")

        audio_files = []

        for instrument_folder in os.listdir(train_dir):
            instrument_dir = os.path.join(train_dir, instrument_folder)

            if os.path.isdir(instrument_dir):
                instrument_tag = instrument_folder

                for file in os.listdir(instrument_dir):
                    if file.endswith(".mp3"):

                        audio, sample_rate = self.load_audio_file(instrument_dir, file)

                        audio_files.append((audio, sample_rate, instrument_tag))

        # Save the list using pickle
        with open(self.ingestion_config.raw_train_path, 'wb') as f:
            pickle.dump(audio_files, f)

        logging.info("Loaded train data without segmentation")



if __name__== '__main__':
    # Create an object of DataIngestion class
    data_ingestion = DataIngestion()
    # Load and save the training audio files
    data_ingestion.load_instruments_train_data()
    # Create an object of DataTransformation class
    data_transformation = DataTransformation()
    # Perform transformation, generate spectrograms, and save to pickle
    data_transformation.transforms()
