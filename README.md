# 🩺 Pneumonia Detection Using CNN

This project is a deep learning-based system for detecting pneumonia in chest X-ray images. Built using TensorFlow and Keras, the CNN model effectively classifies images into "Pneumonia" or "Normal" categories. This tool can assist in medical diagnostics, especially in under-resourced environments.

---

## 🛠️ Technologies Used

- Python 3.x
- TensorFlow / Keras
- NumPy, Pandas
- OpenCV, Matplotlib
- scikit-learn

---

## 📁 Directory Structure

<Br>Dataset Details</Br>
Dataset Name            : Chest X-Ray Images (Pneumonia)
Number of Class         : 2
Number/Size of Images   : Total      : 5856 (1.15 Gigabyte (GB))
                          Training   : 5216 (1.07 Gigabyte (GB))
                          Validation : 320  (42.8 Megabyte (MB))
                          Testing    : 320  (35.4 Megabyte (MB))

Model Parameters
Machine Learning Library: Keras
Base Model              : InceptionV3 && Custom Deep Convolutional Neural Network
Optimizers              : Adam
Loss Function           : categorical_crossentropy

For Custom Deep Convolutional Neural Network : 
Training Parameters
Batch Size              : 64
Number of Epochs        : 30
Training Time           : 2 Hours

Output (Prediction/ Recognition / Classification Metrics)
Testing
Accuracy (F-1) Score    : 89.53%
Loss                    : 0.41
Precision               : 88.37%
Recall (Pneumonia)      : 95.48% (For positive class)
