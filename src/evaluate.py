from logger import logging
import pickle
import numpy as np
import torch.nn.functional as F
import torch
from model import CustomResNet18
from train import convert_labels_to_indices, create_label_mapping
from torch.utils.data import TensorDataset, DataLoader


# Check for Metal GPU support
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device('cpu')

def load_model(model, model_path= "artifacts/cnn_instrument_classifier.pth"):
    logging.info(f"Loading model from {model_path}")
    state_dict = torch.load(model_path, map_location=device)  # Ensure the model is loaded on the correct device
    model.load_state_dict(state_dict)
    model.to(device)  # Move model to the correct device
    model.eval()  # Set model to evaluation mode
    logging.info("Model loaded successfully.")
    return model


def load_filename_label_mapping(filepath):
    filename_label_map = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("File:"):
                # Strip 'File: ' from the beginning and ', Text: ' from the middle to get filename and label
                parts = line.split(", Text: ")
                filename = parts[0].replace("File: ", "").strip()
                label = parts[1].strip()
                
                # Store the filename and label in the dictionary
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
        for inputs, targets in test_loader:
            inputs = inputs.to(device)  # Move the combined input tensor to device
            targets = targets.to(device)

            # Forward pass through the model
            outputs = model(inputs)

            # Get the predicted class by taking the max over the output
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()


    accuracy = correct / total
    return accuracy



def pad_tensor(tensor, target_depth):
    current_rows, current_depth, current_cols = tensor.shape

    # Pad height-wise (the second dimension) if the current number of features is less than target_depth
    if current_depth < target_depth:
        padding = torch.zeros((current_rows, target_depth - current_depth, current_cols), device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=1)  # Concatenate along the second dimension

    return tensor


if __name__ == "__main__":

    with open('artifacts/processed_test_audio_files.pkl', 'rb') as f:
        test_data = pickle.load(f)


    # Load the filename-to-label mapping
    label_mapping_file = 'artifacts/raw_test_audio_files.txt'
    simplified_filename_label_map = load_filename_label_mapping(label_mapping_file)


    labels = []
    spectrograms, mfccs, rmses = [], [], []
    spectral_centroids, spectral_bandwidths = [], []
    spectral_contrasts = []
    spectral_rolloffs, zcrs = [], []
    filenames = []

    for iter in test_data:
        spectrogram = iter['spectrogram']
        filename = iter['filename']
        mfcc = iter['mfcc']
        rmse = iter['rmse']
        spectral_centroid = iter['spectral_centroid']
        spectral_bandwidth = iter['spectral_bandwidth']
        spectral_contrast = iter['spectral_contrast']
        spectral_rolloff = iter['spectral_rolloff']
        zcr = iter['zcr']

        # Strip the '_chunk_{i}' part to get the original filename
        original_filename = filename.split('_chunk')[0]

        # Get the corresponding label from the filename-to-label mapping
        if original_filename in simplified_filename_label_map:
            label = simplified_filename_label_map[original_filename]
            labels.append(label)
            # Store the spectrogram, filename, and label (tag) together
            filenames.append(original_filename)
            spectrograms.append(spectrogram)
            mfccs.append(mfcc)
            rmses.append(rmse)
            spectral_centroids.append(spectral_centroid)
            spectral_bandwidths.append(spectral_bandwidth)
            spectral_contrasts.append(spectral_contrast)
            spectral_rolloffs.append(spectral_rolloff)

        else:
            raise ValueError(f"Filename {filename} (simplified as {filename}) not found in label mapping.")

    target_rows = 130


    mfccs = pad_tensor(mfccs, target_rows)
    rmses = pad_tensor(rmses, target_rows)
    spectral_centroids = pad_tensor(spectral_centroids, target_rows)
    spectral_bandwidths = pad_tensor(spectral_bandwidths, target_rows)
    spectral_contrasts = pad_tensor(spectral_contrasts, target_rows)
    spectral_rolloffs = pad_tensor(spectral_rolloffs, target_rows)
    zcrs = pad_tensor(zcrs, target_rows)


    label_maps = create_label_mapping(labels)

    # Convert the list of labels (tags) to indices if needed
    label_indices = convert_labels_to_indices(labels, label_maps)  # Convert labels (tags) to indices

    labels = torch.tensor(np.array(label_indices), dtype=torch.long).to(device)

    input_tensor = torch.stack([
         spectrograms,
         mfccs,
         rmses,
         spectral_centroids,
         spectral_bandwidths,
         spectral_contrasts,
         spectral_rolloffs,
         zcrs
    ], dim = 1)

    # Create DataLoader for test data using spectrograms_tensor
    test_dataset = TensorDataset(input_tensor,labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load the trained model
    num_classes = len(label_maps)
    model = CustomResNet18(num_classes=num_classes)
    model = load_model(model, 'artifacts/cnn_instrument_classifier.pth')

    # Evaluate the model
    accuracy = evaluate_model(model, test_loader)

    print(f"Test Accuracy: {accuracy * 100:.2f}%")
