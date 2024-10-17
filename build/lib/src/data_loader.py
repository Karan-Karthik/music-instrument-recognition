import os
import numpy as np
import librosa
#from exceptions.DataException import k

def load_audio_file(filepath, sr= 35000):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Audio file not found: {filepath}")
    
    audio, sample_rate = librosa.load(filepath, sr = sr)
    return audio, sample_rate


def load_instruments_data(train_dir):
    audio_files=[]

    for instrument_folder in os.listdir(train_dir):
          instrument_dir = os.path.join(train_dir, instrument_folder)

          if os.path.isdir(instrument_dir):
                instrument_tag = instrument_folder

                for file in os.listdir(instrument_dir):
                      if file.endswith(".wav"):
                            filepath = os.path.join(instrument_dir, file)
                            audio, sample_rate = load_audio_file(filepath)

                            audio_files.append((audio, sample_rate, instrument_tag))

    return audio_files


def convert_to_spectrogram(audio, sample_rate, n_mels= 130):
     spectrogram= librosa.feature.melspectrogram(y= audio, sr= sample_rate, n_mels= n_mels)
     return librosa.power_to_db(spectrogram, ref=np.max)


