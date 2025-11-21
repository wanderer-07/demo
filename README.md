ISL Recognition System Implementation Plan
Goal Description
Build a real-time Indian Sign Language (ISL) recognition system using Python and Convolutional Neural Networks (CNN). The system will take video input from a webcam, process the frames, and predict the corresponding sign language gesture.

User Review Required
IMPORTANT

Dataset Extraction: The dataset is currently inside a RAR file (ISL custom Data.rar). I was unable to extract it automatically. Please extract this RAR file into c:/Users/Parag K B/ISL/dataset/ so that the class folders are visible.

Proposed Changes
Data Processing
[NEW] 
data_loader.py
Script to load images from the dataset directory.
Use ImageDataGenerator for augmentation (rescaling, rotation, zoom) to improve model robustness.
Split data into training and validation sets.
Model Architecture
[NEW] 
model.py
Define a CNN architecture using TensorFlow/Keras.
Layers:
Conv2D layers with ReLU activation.
MaxPooling2D for downsampling.
Dropout for regularization.
Flatten and Dense layers for classification.
Output layer with Softmax activation (size = number of classes).
Training
[NEW] 
train.py
Script to compile and train the model.
Save the best model weights to isl_model.h5.
Plot training history (accuracy/loss).
Real-time Inference
[NEW] 
realtime_detection.py
Use OpenCV (cv2) to capture video from the webcam.
Preprocess frames (resize, normalize) to match model input.
Predict the class using the trained model.
Display the predicted label on the video feed.
Verification Plan
Automated Tests
Model Summary: Run model.summary() to verify architecture.
Training Dry Run: Run training for 1 epoch to ensure pipeline works.
Manual Verification
Real-time Test: Run realtime_detection.py and perform gestures in front of the camera to verify predictions.
