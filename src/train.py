import pickle
import torch
import numpy as np
from model import SpectrogramResNet
from logger import logging
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

def create_label_mapping(labels):
    unique_labels = sorted(set(labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    return label_to_index

def convert_labels_to_indices(labels, label_to_index):
    return [label_to_index[label] for label in labels]

def save_model(model, filename='/Users/karanwork/Documents/Deep learning project/music-instrument-recognition/artifacts/cnn_instrument_classifier.pth'):
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

        for batch_idx, (inputs, targets) in enumerate(train_loader):
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
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        # Validation loop after each epoch
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
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



if __name__ == "__main__":
    with open('/Users/karanwork/Documents/Deep learning project/music-instrument-recognition/artifacts/processed_train_audio_files.pkl', 'rb') as f:
        train_data = pickle.load(f)

    spectrograms = []
    labels = []

    for iter in train_data:
        spectrograms.append(iter['spectrogram'])
        labels.append(iter['instrument_tag'])

    label_maps = create_label_mapping(labels)
    label_indices = convert_labels_to_indices(labels, label_maps)

    spectrograms = np.array(spectrograms)

    spectrograms = torch.tensor(spectrograms, dtype=torch.float32)

    labels = torch.tensor(np.array(label_indices), dtype=torch.long)
    
    # Use resized spectrograms for the dataset
    data = TensorDataset(spectrograms, labels)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_dataset, val_dataset = random_split(data, [train_size, val_size])

    # Create DataLoaders for batching
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    num_class = len(label_maps)

    model = SpectrogramResNet(num_classes= num_class)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_validate_model(model, train_loader, val_loader,criterion, optimizer, num_epochs = 20)
