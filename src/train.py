import pickle
import torch
import numpy as np
from model import MultiInputCNN, CustomResNet18
from logger import logging
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


# Check for Metal GPU support
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("mps")


def create_label_mapping(labels):
    unique_labels = sorted(set(labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    return label_to_index


def convert_labels_to_indices(labels, label_to_index):
    return [label_to_index[label] for label in labels]


def save_model(model, filename='artifacts/cnn_instrument_classifier.pth'):
    logging.info(f"Saving model to {filename}")
    torch.save(model.state_dict(), filename)
    logging.info("Model saved successfully.")


def train_validate_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        # Training loop
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)  # Move the combined input tensor to device
            targets = targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Calculate running loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        # Print epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.2f}%")

        # Validation loop after each epoch
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch_data in val_loader:
                inputs, targets = batch_data  # Unpack the combined input tensor and targets
                inputs = inputs.to(device)  # Move the combined input tensor to device
                targets = targets.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100 * correct_val / total_val
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

    save_model(model)


def normalize(tensor):
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / (std + 1e-6)  # Adding a small value to avoid division by zero


def pad_tensor(tensor, target_depth):
    current_rows, current_depth, current_cols = tensor.shape

    # Pad height-wise (the second dimension) if the current number of features is less than target_depth
    if current_depth < target_depth:
        padding = torch.zeros((current_rows, target_depth - current_depth, current_cols), device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=1)  # Concatenate along the second dimension

    return tensor


# Data loading and preprocessing
if __name__ == "__main__":
    with open('artifacts/processed_train_audio_files.pkl', 'rb') as f:
        train_data = pickle.load(f)

    # Initialize lists for each feature and labels
    labels = []
    spectrograms, mfccs, rmses = [], [], []
    spectral_centroids, spectral_bandwidths = [], []
    spectral_contrasts = []
    spectral_rolloffs, zcrs = [], []

    # Extract features and labels
    for item in train_data:
        labels.append(item['instrument_tag'])
        spectrograms.append(item['spectrogram'])
        mfccs.append(item['mfcc'])
        rmses.append(item['rmse'])
        spectral_centroids.append(item['spectral_centroid'])
        spectral_bandwidths.append(item['spectral_bandwidth'])
        spectral_contrasts.append(item['spectral_contrast'])
        spectral_rolloffs.append(item['spectral_rolloff'])
        zcrs.append(item['zcr'])


    target_rows = 130
    
    spectrogram_tensors = normalize(torch.tensor(np.array(spectrograms), dtype=torch.float32, device= device)), target_rows
    mfcc_tensors = pad_tensor(normalize(torch.tensor(np.array(mfccs), dtype=torch.float32, device=device)), target_rows)
    rmse_tensors = pad_tensor(normalize(torch.tensor(np.array(rmses), dtype=torch.float32, device=device)), target_rows)
    spectral_centroid_tensors = pad_tensor(normalize(torch.tensor(np.array(spectral_centroids), dtype=torch.float32, device=device)), target_rows)
    spectral_bandwidth_tensors = pad_tensor(normalize(torch.tensor(np.array(spectral_bandwidths), dtype=torch.float32, device=device)), target_rows)
    spectral_contrast_tensors = pad_tensor(normalize(torch.tensor(np.array(spectral_contrasts), dtype=torch.float32, device=device)), target_rows)
    spectral_rolloff_tensors = pad_tensor(normalize(torch.tensor(np.array(spectral_rolloffs), dtype=torch.float32, device=device)), target_rows)
    zcr_tensors = pad_tensor(normalize(torch.tensor(np.array(zcrs), dtype = torch.float32, device=device)), target_rows)


    input_tensor = torch.stack([
         spectrogram_tensors,
         mfcc_tensors,
         rmse_tensors,
         spectral_centroid_tensors,
         spectral_bandwidth_tensors,
         spectral_contrast_tensors,
         spectral_rolloff_tensors,
         zcr_tensors
    ], dim = 1)

    # Convert labels to indices
    label_indices = convert_labels_to_indices(labels, create_label_mapping(labels))
    labels_tensor = torch.tensor(label_indices, dtype=torch.long)

    # Create TensorDataset
    data = TensorDataset(input_tensor, labels_tensor)

    # Split into training and validation sets
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_dataset, val_dataset = random_split(data, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    # Model, criterion, optimizer
    num_classes = len(create_label_mapping(labels))
    model = CustomResNet18(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    # Train the model
    train_validate_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=15)
