# Music Instrument Recognition using SVM with RBF Kernel

This project aims to classify instruments played in a given audio input using Support Vector Machines (SVM) with the Radial Basis Function (RBF) kernel. The feature extraction process involves using Mel Frequency Cepstral Coefficients (MFCCs) to represent audio data as input features for training and classification.

The model achieved high classification performance, demonstrating its effectiveness in instrument recognition tasks. Specifically, it achieved an accuracy of 99.17% on the test data.

## MFCC Extraction

**MFCCs** (Mel Frequency Cepstral Coefficients) are features that represent the short-term power spectrum of sound, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency. In this project, MFCCs are used as input features to classify the audio into different instrument categories.


## SVM with RBF Kernel

### How SVM Works:
**Support Vector Machine (SVM)** is a supervised learning algorithm commonly used for classification tasks. It works by finding a hyperplane in a high-dimensional space that separates the data points of different classes with the maximum margin. The RBF kernel (Radial Basis Function) is a popular kernel used in SVM, which maps the input space into a higher-dimensional space where linear separation is possible. It works well for non-linearly separable data.


![SVM Visualization](images/support-vector-machine-svm.jpg.webp)

![SVM with RBF kernel](images/rbf_kernel.png)



## Model Performance

### Key Metrics:

- **Accuracy**: 99.1667%
- **Precision**: 99.2222%
- **Recall**: 99.1667%
- **F1-Score**: 99.1698%

### Detailed Classification Report:

| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Cello     | 1.00      | 1.00   | 1.00     | 27      |
| Flute     | 1.00      | 1.00   | 1.00     | 19      |
| Oboe      | 1.00      | 1.00   | 1.00     | 20      |
| Saxophone | 1.00      | 0.95   | 0.97     | 19      |
| Trumpet   | 1.00      | 1.00   | 1.00     | 21      |
| Viola     | 0.93      | 1.00   | 0.97     | 14      |
| **Overall Accuracy** |       |        | **0.99** | 120     |


Below is a heatmap illustrating the classification performance, where a value of 1 indicates 100% accuracy.

![SVM with RBF kernel](images/classification_report_heatmap.png)

This report demonstrates the high performance of the model on the validation set. High accuracy, precision, and recall across different instrument classes reflect the effectiveness of the feature extraction and the SVM classifier.


---

## How to Run the Project

Follow these steps to set up and run the project:

### Step 1: Install Dependencies
Ensure all required dependencies are installed by running the following command in the project directory:

```bash
pip install -e .
```

This will install all necessary libraries, including `librosa`, `torch`, and `scikit-learn`.

---

### Step 2: Load and Transform Data
The `data_loader.py` script loads raw audio files, preprocesses them into features such as spectrograms and MFCCs, and saves the processed data into a pickle file.

Run the following command to execute the data loader and transform the data:

```bash
python src/data_loader.py
```

After running this script, the processed data will be saved in the `artifacts` folder as `processed_train_audio_files.pkl`.

---

### Step 3: Train the Model
The `train_svm.py` script trains the SVM classifier. It:
1. Loads the processed data from the pickle file.
2. Splits it into training and validation sets.
3. Trains the SVM model using an RBF kernel.
4. Saves the trained model as a pickle file.

Run the training script with the following command:

```bash
python src/train_svm.py
```

After training, the model will be saved as `artifacts/svm_model.pkl`.

---

### Step 4: Evaluate the Model
After training, the `train_svm.py` script generates:
- A detailed classification report, including precision, recall, F1-score, and overall accuracy.
- A heatmap illustrating the classification performance for each class, saved as an image (e.g., `images/classification_report_heatmap.png`).

Check the console output for metrics and review the saved heatmap image for visual insights.

---

### Output Files
1. **Processed Data**:
   - `artifacts/processed_train_audio_files.pkl`
2. **Trained Model**:
   - `artifacts/svm_model.pkl`
3. **Performance Metrics**:
   - Classification heatmap saved as an image.

---

