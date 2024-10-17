from logger import logging
import os
import pickle
import numpy as np
import torch
from model import SpectrogramResNet
from train import convert_labels_to_indices, create_label_mapping
from torch.utils.data import TensorDataset, DataLoader

def load_model(model, model_path= "/Users/karanwork/Documents/Deep learning project/music-instrument-recognition/artifacts/cnn_instrument_classifier.pth"):
    logging.info(f"Loading model from {model_path}")
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()  # Set model to evaluation mode
    logging.info("Model loaded successfully.")
    return model

# Function to pad spectrograms to the target width
def pad_spectrogram(spectrogram, target_width):
        current_width = spectrogram.shape[-1]
        if current_width < target_width:
            pad_width = target_width - current_width
            # Padding along the last dimension (width) with a constant value (e.g., -161 for silence)
            padded_spectrogram = np.pad(spectrogram, ((0, 0), (0, 0), (0, pad_width)), mode='constant', constant_values=-161)
            return padded_spectrogram
        return spectrogram 


# Function to simplify filenames for better matching
def simplify_filename(filename):
    # Lowercase, remove spaces, file extensions, and special characters
    return os.path.splitext(filename.lower().replace(" ", "").replace("-", "").replace("_", ""))[0]

def load_filename_label_mapping(filepath):
    filename_label_map = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("File:"):
                parts = line.split(", Text: ")
                filename = parts[0].replace("File: ", "").strip()
                label = parts[1].strip()
                filename_label_map[filename] = label
    return filename_label_map

def evaluate_model(model, test_loader):
    """
    Evaluates the trained model on the test dataset and returns the accuracy.
    """
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Turn off gradients for evaluation
        for inputs, labels in test_loader:

            # Forward pass through the model
            outputs = model(inputs)

            # Get the predicted class by taking the max over the output
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy


if __name__ == "__main__":

    with open('artifacts/processed_test_audio_files.pkl', 'rb') as f:
        test_data = pickle.load(f)


    # Load the filename-to-label mapping
    label_mapping_file = '/Users/karanwork/Documents/Deep learning project/music-instrument-recognition/artifacts/raw_test_audio_files.txt'
    filename_label_map = load_filename_label_mapping(label_mapping_file)

    spectrograms = []
    labels = []
    filenames = []

    # Simplify the keys in the label mapping for better matching
    simplified_filename_label_map = {simplify_filename(k): v for k, v in filename_label_map.items()}
    spectrogram_info = []
    

    for iter in test_data:
        spectrogram = iter['spectrogram']
        filename = iter['filename']
        filenames.append(filename)
        spectrograms.append(spectrogram)

        # Simplify the filename for better matching
        simplified_filename = simplify_filename(filename)

        # Get the corresponding label from the filename-to-label mapping
        if simplified_filename in simplified_filename_label_map:
            label = simplified_filename_label_map[simplified_filename]
            labels.append(label)

            # Store the spectrogram, filename, and label (tag) together
            spectrogram_info.append({
                'filename': filename,
                'spectrogram': spectrogram,
                'label': label
            })
        else:
            raise ValueError(f"Filename {filename} (simplified as {simplified_filename}) not found in label mapping.")


    # Extract spectrograms and labels from spectrogram_info
    spectrograms = [info['spectrogram'] for info in spectrogram_info]  # Extract the spectrogram tensors
    labels = [info['label'] for info in spectrogram_info]  # Extract the corresponding labels (tags)

    label_maps = create_label_mapping(labels)

    # Convert the list of labels (tags) to indices if needed
    # Assuming you have already mapped your labels to indices (label_indices)
    label_indices = convert_labels_to_indices(labels, label_maps)  # Convert labels (tags) to indices
    

    # Target width based on the maximum size used during training
    target_width = 1723
    # Apply padding to all spectrograms that are smaller than the target width
    padded_spectrograms = [pad_spectrogram(spectrogram, target_width) for spectrogram in spectrograms]

    spectrograms_k = np.array(padded_spectrograms)
    spectrograms = torch.tensor(spectrograms_k, dtype=torch.float32)
    labels = torch.tensor(np.array(label_indices), dtype=torch.long)

    # Create DataLoader for test data
    test_dataset = TensorDataset(spectrograms, labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load the trained model
    num_classes = len(label_maps)
    model = SpectrogramResNet(num_classes=num_classes)
    model = load_model(model, '/Users/karanwork/Documents/Deep learning project/music-instrument-recognition/artifacts/cnn_instrument_classifier.pth')

    # Evaluate the model
    accuracy = evaluate_model(model, test_loader)

    print(f"Test Accuracy: {accuracy * 100:.2f}%")
