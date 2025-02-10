import cv2
import os
import numpy as np
import glob
from tqdm import tqdm
import mediapipe as mp
from imquality import brisque
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def extract_features(video_path):
    """extract features from a video, traditional CV techniques"""

    cap = cv2.VideoCapture(video_path)

    features = []
    
    success, frame = cap.read()
    if not success:
        return None
    
    # grayscale features
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    features.extend([
        np.mean(gray), 
        np.std(gray),   
        np.max(gray),   
        np.min(gray),   
    ])
    
    # histogram features
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten()
    hist = hist / hist.sum()  
    features.extend(hist)
    
    # Haralick texture features
    glcm = cv2.GaussianBlur(gray, (5,5), 0)
    features.extend([
        cv2.Laplacian(glcm, cv2.CV_64F).var(), 
        cv2.Sobel(glcm, cv2.CV_64F, 1, 0, ksize=5).var(),  
        cv2.Sobel(glcm, cv2.CV_64F, 0, 1, ksize=5).var() 
    ])
    
    cap.release()
    return np.array(features)

def process_dataset(base_path):
    """process all videos, prepare for training, validation and testing."""
    X_train, y_train = [], []  
    X_val, y_val = [], []       
    X_test, y_test = [], []    
    
    # Process training data
    train_path = os.path.join(base_path, 'train', '*.mp4')
    for video_path in tqdm(glob.glob(train_path), desc="Processing training data"):
        filename = os.path.basename(video_path)
        label = 1 if filename.startswith('id') else 0
        features = extract_features(video_path)
        if features is not None:
            X_train.append(features)
            y_train.append(label)

    # Process validation data
    val_path = os.path.join(base_path, 'val', '*.mp4')
    for video_path in tqdm(glob.glob(val_path), desc="Processing validation data"):
        filename = os.path.basename(video_path)
        label = 1 if filename.startswith('id') else 0
        features = extract_features(video_path)
        if features is not None:
            X_val.append(features)
            y_val.append(label)
    
    # Process test data
    test_path = os.path.join(base_path, 'test', '*.mp4')
    for video_path in tqdm(glob.glob(test_path), desc="Processing test data"):
        filename = os.path.basename(video_path)
        label = 1 if filename.startswith('id') else 0
        features = extract_features(video_path)
        if features is not None:
            X_test.append(features)
            y_test.append(label)
    
    return (np.array(X_train), np.array(X_val), np.array(X_test),
            np.array(y_train), np.array(y_val), np.array(y_test))

def train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test):
    """train and evaluate model"""

    model = XGBClassifier(eval_metric='logloss', early_stopping_rounds=10)  
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\nTest Set Metrics:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}") 
    print(f"F1 Score: {f1:.3f}")
    
    return model

def predict_non_dl_UI(model, vid):
    """predict whether a single video is real or fake for UI implementation"""
    features = extract_features(vid)
    if features is not None:
        prediction = model.predict([features])[0]
        return "Fake" if prediction == 1 else "Real"
    return "Error processing video"

if __name__ == "__main__":

    base_path = "Celeb-DF" 
    
    X_train, X_test, X_val, y_train, y_val, y_test = process_dataset(base_path)
    model = train_and_evaluate(X_train,X_test,X_val,y_train,y_val,y_test)