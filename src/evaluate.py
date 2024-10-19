from logger import logging
import os
import pickle
import numpy as np
import torch.nn.functional as F
import torch
from model import CustomCNN, YOLOLikeCNN
from train import convert_labels_to_indices, create_label_mapping
from torch.utils.data import TensorDataset, DataLoader

# Check for Metal GPU support
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_model(model, model_path= "/Users/karanwork/Documents/Deep learning project/music-instrument-recognition/artifacts/cnn_instrument_classifier.pth"):
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
        for inputs, labels in test_loader:
            # Move inputs and labels to the device (CPU or GPU)
            inputs, labels = inputs.to(device), labels.to(device)

            # Ensure inputs have 1 channel (grayscale)
            if inputs.shape[1] != 1:
                inputs = inputs.unsqueeze(1)  # Add the channel dimension if missing

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
    simplified_filename_label_map = load_filename_label_mapping(label_mapping_file)

    spectrograms = []
    labels = []
    filenames = []
    spectrogram_info = []
    

    for iter in test_data:
        spectrogram = iter['spectrogram']
        filename = iter['filename']
        filenames.append(filename)
        spectrograms.append(spectrogram)

        # Get the corresponding label from the filename-to-label mapping
        if filename in simplified_filename_label_map:
            label = simplified_filename_label_map[filename]
            labels.append(label)

            # Store the spectrogram, filename, and label (tag) together
            spectrogram_info.append({
                'filename': filename,
                'spectrogram': spectrogram,
                'label': label
            })
        else:
            raise ValueError(f"Filename {filename} (simplified as {filename}) not found in label mapping.")


    # Extract spectrograms and labels from spectrogram_info
    spectrograms = [info['spectrogram'] for info in spectrogram_info]  # Extract the spectrogram tensors
    labels = [info['label'] for info in spectrogram_info]  # Extract the corresponding labels (tags)

    label_maps = create_label_mapping(labels)

    # Convert the list of labels (tags) to indices if needed
    label_indices = convert_labels_to_indices(labels, label_maps)  # Convert labels (tags) to indices
    
    # Target width based on the maximum size used during training
    target_width = 259

    # Assuming spectrograms is a list of numpy arrays (1, 130, variable_width)
    # First, convert all spectrograms to a PyTorch tensor and interpolate them to the target width
    padded_spectrograms = []
    for spectrogram in spectrograms:
        #print(f"Original shape: {spectrogram.shape}")
        
        # Convert numpy array to tensor and add channel dimension to make it 4D (N, C, H, W)
        spectrogram_tensor = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)  # Assuming it’s already 3D, we add one more dimension for the batch
        
        # Interpolate to resize the width to the target width (259), keeping height fixed at 130
        resized_spectrogram = F.interpolate(spectrogram_tensor, size=(130, target_width), mode='bilinear', align_corners=False)
        
        # Append the resized spectrogram to the list (removing the batch dimension if needed)
        padded_spectrograms.append(resized_spectrogram.squeeze(0))
        #print(f"Resized shape: {resized_spectrogram.shape}")

    # Concatenate and move the tensor to the correct device
    spectrograms_tensor = torch.cat(padded_spectrograms, dim=0).to(device)

    labels = torch.tensor(np.array(label_indices), dtype=torch.long).to(device)


    # Create DataLoader for test data using spectrograms_tensor
    test_dataset = TensorDataset(spectrograms_tensor.squeeze(1), labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load the trained model
    num_classes = len(label_maps)
    model = YOLOLikeCNN(num_classes=num_classes)
    model = load_model(model, '/Users/karanwork/Documents/Deep learning project/music-instrument-recognition/artifacts/cnn_instrument_classifier.pth')

    # Evaluate the model
    accuracy = evaluate_model(model, test_loader)

    print(f"Test Accuracy: {accuracy * 100:.2f}%")
