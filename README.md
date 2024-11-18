# Music Instrument Recognition using SVM with RBF Kernel

This project aims to classify instruments played in a given audio input using Support Vector Machines (SVM) with the Radial Basis Function (RBF) kernel. The feature extraction process involves using Mel Frequency Cepstral Coefficients (MFCCs) to represent audio data as input features for training and classification.

The model achieved high classification performance, demonstrating its effectiveness in instrument recognition tasks. Specifically, it achieved an accuracy of 98.33% on the test data.

## MFCC Extraction

**MFCCs** (Mel Frequency Cepstral Coefficients) are features that represent the short-term power spectrum of sound, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency. In this project, MFCCs are used as input features to classify the audio into different instrument categories.


## SVM with RBF Kernel

### How SVM Works:
**Support Vector Machine (SVM)** is a supervised learning algorithm commonly used for classification tasks. It works by finding a hyperplane in a high-dimensional space that separates the data points of different classes with the maximum margin. The RBF kernel (Radial Basis Function) is a popular kernel used in SVM, which maps the input space into a higher-dimensional space where linear separation is possible. It works well for non-linearly separable data.




## Model Performance

### Key Metrics:

- **Accuracy**: 98.33%
- **Precision**: 98.49%
- **Recall**: 98.33%
- **F1-Score**: 98.35%

### Detailed Classification Report:

| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Cello     | 1.00      | 1.00   | 1.00     | 27      |
| Flute     | 1.00      | 1.00   | 1.00     | 14      |
| Oboe      | 1.00      | 1.00   | 1.00     | 20      |
| Saxophone | 0.90      | 1.00   | 0.95     | 19      |
| Trumpet   | 1.00      | 0.95   | 0.97     | 19      |
| Viola     | 1.00      | 0.95   | 0.98     | 21      |
| **Overall Accuracy** |       |        | **0.98** | 120     |


This report demonstrates the high performance of the model on the validation set. High accuracy, precision, and recall across different instrument classes reflect the effectiveness of the feature extraction and the SVM classifier.
