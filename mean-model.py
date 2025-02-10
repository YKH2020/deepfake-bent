import os
import glob
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import cv2
import numpy as np

def extract_features(video_path, num_frames=5):
    """Extract simple mean-based features for deepfake detection using 5 sampled frames."""

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)  # Evenly spaced frames
    frame_means = []
    frame_stds = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # Jump to the selected frame
        success, frame = cap.read()
        if not success:
            continue  # Skip if frame read fails
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        frame_means.append(np.mean(gray))
        frame_stds.append(np.std(gray))

    cap.release()

    return 0.5 * np.mean(frame_means) + 0.5 * np.mean(frame_stds)  

def process_dataset(base_path):
    """process videos in the dataset and prepare for training, validation and testing"""
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    # Process training data
    train_path = os.path.join(base_path, 'train', '*.mp4')
    for video_path in tqdm(glob.glob(train_path), desc="Processing training data"):
        filename = os.path.basename(video_path)
        label = 1 if (filename.startswith('id') and 
              len(filename) > 2 and 
              filename[2].isdigit() and 
              filename[3:6] == '_id') else 0
        features = extract_features(video_path)
        if features is not None:
            X_train.append(features)
            y_train.append(label)
    
    # Process test data
    test_path = os.path.join(base_path, 'test', '*.mp4')
    for video_path in tqdm(glob.glob(test_path), desc="Processing test data"):
        filename = os.path.basename(video_path)
        label = 1 if (filename.startswith('id') and 
              len(filename) > 2 and 
              filename[2].isdigit() and 
              filename[3:6] == '_id') else 0
        features = extract_features(video_path)
        if features is not None:
            X_test.append(features)
            y_test.append(label)
    
    return (np.array(X_train), np.array(X_test),
            np.array(y_train), np.array(y_test))

def train_mean_model(X_train, y_train):
    """train a simple mean threshold model"""
    real_means = [X_train[i] for i in range(len(X_train)) if y_train[i] == 0]
    fake_means = [X_train[i] for i in range(len(X_train)) if y_train[i] == 1]
    
    threshold = (np.mean(real_means) + np.mean(fake_means)) / 2
    
    return threshold

def predict_mean_model(X, threshold):
    """predict labels based on the threshold"""
    return np.array([1 if x >= threshold else 0 for x in X])

def predict_mean_model_UI(vid, threshold):
    """prediction for UI implementation"""
    x = extract_features(vid)
    return 'Fake' if x >= threshold else 'Real'

if __name__ == "__main__":
    """train and evaluate the mean model"""

    base_path = "Celeb-DF" 
    X_train, X_test, y_train, y_test = process_dataset(base_path)
    
    threshold = train_mean_model(X_train, y_train)

    # see threshold to set for UI
    print(f"Threshold: {threshold}")
    
    preds = predict_mean_model(X_test, threshold)
    
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    print("\nTest Set Metrics:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}") 
    print(f"F1 Score: {f1:.3f}")
   