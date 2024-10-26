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

def resize_tensor(tensor, target_size):
            # Resize tensor to match target_size (height and width)
        return F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)


def evaluate_model(model, test_loader):
    """
    Evaluates the trained model on the test dataset and returns the accuracy.
    """
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    
    with torch.no_grad():  # Turn off gradients for evaluation
        for spectrograms, mfccs, rmses, targets in test_loader:
             # Debug print to see the shape of inputs
            print(f"Shape of spectrograms: {spectrograms.shape}")
            print(f"Shape of mfccs: {mfccs.shape}")
            print(f"Shape of rmses: {rmses.shape}")
            # Move inputs and labels to the device (CPU or GPU)
            spectrograms, mfccs, rmses, targets = spectrograms.to(device), mfccs.to(device), rmses.to(device), targets.to(device)

            # Ensure mfccs and rmses have the same number of dimensions as spectrograms (4D)
            if mfccs.dim() == 3:
                mfccs = mfccs.unsqueeze(1)  # Add a channel dimension to make it 4D
            if rmses.dim() == 2:
                rmses = rmses.unsqueeze(1)  # Add a channel dimension to make it 4D
            if spectrograms.dim() == 3:
                spectrograms = spectrograms.unsqueeze(1)
            
            print(f"Shape of spectrograms: {spectrograms.shape}")
            # Check the shapes again after unsqueeze
            print(f"Resized shape of rmses: {rmses.shape}")


            
            #mfccs = resize_tensor(mfccs, target_size)
            #rmses = resize_tensor(rmses, target_size)
         #mfccs_resized = F.interpolate(mfccs.unsqueeze(1), size=target_size, mode='bilinear', align_corners=False).squeeze(1)    
            # Option 1: Concatenate spectrograms, mfccs, and rmses along the channel dimension

            # Calculate how much padding is needed for the height
            padding_amount = spectrograms.shape[2] - mfccs.shape[1]  # 130 (spectrogram height) - 128 (mfcc height)

            # Only pad if padding is needed
            if padding_amount > 0:
                mfccs_padded = F.pad(mfccs.unsqueeze(1), (0, 0, 0, padding_amount), mode='constant', value=0).squeeze(1)
            elif padding_amount < 0:
                # Truncate mfccs if it's larger than the target size
                mfccs_padded = mfccs[:, :spectrograms.shape[2], :]  # Truncate to match the height of 130
            else:
                mfccs_padded = mfccs  # No padding if sizes already match

            """
            # Calculate how much padding is needed for the height
            padding_amount = spectrograms.shape[2] - rmse.shape[1]  # 130 (spectrogram height) - 128 (mfcc height)

            # Only pad if padding is needed
            if padding_amount > 0:
                # Pad the height (frequency bins) dimension; no padding for width
                # Pad on both sides equally, so pad (0, 0) for width and pad (0, padding_amount) for height
                mfccs_padded = F.pad(mfccs.unsqueeze(1), (0, 0, 0, padding_amount), mode='constant', value=0).squeeze(1)
            else:
                mfccs_padded = mfccs  # No padding if sizes already match
            """ 

            print(f"Shape of spectrograms: {spectrograms.shape}")
            # Check the shapes again after unsqueeze
            print(f"Shape of mfccs: {mfccs_padded.shape}")
            print(f"Resized shape of rmses: {rmses.shape}")
            

            inputs = torch.cat((spectrograms, mfccs_padded, rmses), dim=1)  # Assuming inputs should be concatenated

            # Forward pass through the model
            outputs = model(inputs)

            # Get the predicted class by taking the max over the output
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()


    accuracy = correct / total
    return accuracy

def pad_mfcc(mfcc, target_shape):
    # Pad the tensor to match the target shape (height, width)
        padding = (0, target_shape[-1] - mfcc.shape[-1])  # Only pad the width (second dimension)
        return torch.nn.functional.pad(mfcc, padding, "constant", 0)

def pad_rmse(rmse, target_shape):
    # Pad the tensor to match the target shape (height, width)
        padding = (0, target_shape[-1] - rmse.shape[-1])  # Only pad the width (second dimension)
        return torch.nn.functional.pad(rmse, padding, "constant", 0)


if __name__ == "__main__":

    with open('artifacts/processed_test_audio_files.pkl', 'rb') as f:
        test_data = pickle.load(f)


    # Load the filename-to-label mapping
    label_mapping_file = '/Users/karanwork/Documents/Deep learning project/music-instrument-recognition/artifacts/raw_test_audio_files.txt'
    simplified_filename_label_map = load_filename_label_mapping(label_mapping_file)

    spectrograms = []
    mfccs = []
    rmses = []
    labels = [] 
    filenames = []
    spectrogram_info = []
    

    for iter in test_data:
        spectrogram = iter['spectrogram']
        filename = iter['filename']
        mfcc = iter['mfcc']
        rmse = iter['rmse']    

        # Strip the '_chunk_{i}' part to get the original filename
        original_filename = filename.split('_chunk')[0]

        # Get the corresponding label from the filename-to-label mapping
        if original_filename in simplified_filename_label_map:
            label = simplified_filename_label_map[original_filename]
            labels.append(label)
            # Store the spectrogram, filename, and label (tag) together
            spectrogram_info.append({
                'filename': filename,
                'spectrogram': spectrogram,
                'mfcc': mfcc,
                'rmse': rmse,
                'label': label
            })
        else:
            raise ValueError(f"Filename {filename} (simplified as {filename}) not found in label mapping.")


    # Extract spectrograms and labels from spectrogram_info
    spectrograms = [info['spectrogram'] for info in spectrogram_info]  # Extract the spectrogram tensors

    labels = [info['label'] for info in spectrogram_info]  # Extract the corresponding labels (tags)
    mfccs = [info['mfcc'] for info in spectrogram_info]
    rmses = [info['rmse'] for info in spectrogram_info]

    label_maps = create_label_mapping(labels)

    # Convert the list of labels (tags) to indices if needed
    label_indices = convert_labels_to_indices(labels, label_maps)  # Convert labels (tags) to indices


    """    
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

    # Find the maximum width among the mfccs
    max_width = max(mfcc.shape[-1] for mfcc in mfccs)

    # Pad each mfcc to have the same width
    padded_mfccs = [pad_mfcc(mfcc, (mfcc.shape[0], max_width)) for mfcc in mfccs]

    # Stack them into a single tensor
    mfccs = torch.stack(padded_mfccs).to(device)

    # Find the maximum width among the mfccs
    max_width = max(rmse.shape[-1] for rmse in rmses)

    # Pad each mfcc to have the same width
    padded_rmses = [pad_rmse(rmse, (rmse.shape[0], max_width)) for rmse in rmses]

    # Stack them into a single tensor
    rmses = torch.stack(padded_rmses).to(device)
"""
    #mfccs = torch.tensor(mfccs, dtype=torch.float32).to(device)
    #rmses = torch.tensor(rmses, dtype=torch.float32).to(device)


    # Find the maximum length of the spectrograms
    max_length = max(spectrogram.shape[2] for spectrogram in spectrograms)

    # Pad each spectrogram to the maximum length
    padded_spectrograms = []
    for spectrogram in spectrograms:
        pad_width = max_length - spectrogram.shape[2]
        padded_spectrogram = np.pad(spectrogram, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
        padded_spectrograms.append(padded_spectrogram)

    # Convert to a NumPy array
    spectrograms = np.array(padded_spectrograms)

    spectrograms = torch.tensor(spectrograms, dtype=torch.float32).to(device)

    # Find the maximum width among the mfccs
    max_width = max(mfcc.shape[-1] for mfcc in mfccs)

    # Pad each mfcc to have the same width
    padded_mfccs = [pad_mfcc(mfcc, (mfcc.shape[0], max_width)) for mfcc in mfccs]

    # Stack them into a single tensor
    mfccs = torch.stack(padded_mfccs).to(device)

    # Find the maximum width among the mfccs
    max_width = max(rmse.shape[-1] for rmse in rmses)

    # Pad each mfcc to have the same width
    padded_rmses = [pad_rmse(rmse, (rmse.shape[0], max_width)) for rmse in rmses]

    # Stack them into a single tensor
    rmses = torch.stack(padded_rmses).to(device)

    labels = torch.tensor(np.array(label_indices), dtype=torch.long).to(device)

    # Create DataLoader for test data using spectrograms_tensor
    test_dataset = TensorDataset(spectrograms, mfccs, rmses,labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load the trained model
    num_classes = len(label_maps)
    model = YOLOLikeCNN(num_classes=num_classes)
    model = load_model(model, '/Users/karanwork/Documents/Deep learning project/music-instrument-recognition/artifacts/cnn_instrument_classifier.pth')

    # Evaluate the model
    accuracy = evaluate_model(model, test_loader)

    print(f"Test Accuracy: {accuracy * 100:.2f}%")
