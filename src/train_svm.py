import os
from logger import logging
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

# Load Data
with open('artifacts/processed_train_audio_files.pkl', 'rb') as f:
    train_data = pickle.load(f)

# Initialize lists for features and labels
labels = []
features = []

# Extract features and labels
for item in train_data:
    labels.append(item['instrument_tag'])
    combined_features = np.concatenate([item['mfcc'].flatten()])
    features.append(combined_features)

# Convert to NumPy arrays
features = np.array(features)
labels = np.array(labels)

# Encode labels as integers if needed
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Standardize features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=80085)

# Train the model
svm_model = SVC(kernel='rbf', C=15, gamma='scale', class_weight='balanced')

svm_model.fit(X_train, y_train)

# Predict on the validation set
y_pred = svm_model.predict(X_val)

# Calculate and display statistics
accuracy = accuracy_score(y_val, y_pred) * 100  # Convert to percentage
precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted')

# Display metrics in percentages with more precision
print(f"Accuracy: {accuracy:.4f}%")
print(f"Precision: {precision * 100:.4f}%")
print(f"Recall: {recall * 100:.4f}%")
print(f"F1-Score: {f1 * 100:.4f}%")
print("\nDetailed Classification Report:")
print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))


# Save the model
artifacts_folder = 'artifacts'
model_filename = 'svm_model.pkl'

model_path = os.path.join(artifacts_folder, model_filename)

try:
    with open(model_path, 'wb') as file:
        pickle.dump(svm_model, file)
    logging.info(f"Model successfully saved to {model_path}")
except Exception as e:
    logging.error(f"Error saving model: {e}")
