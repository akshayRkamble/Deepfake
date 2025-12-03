import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def create_multimodal_dataset():
    """Create sample multimodal dataset"""
    np.random.seed(42)
    n_samples = 500
    
    # Image features (simulated)
    image_features = np.random.randn(n_samples, 64)
    
    # Audio features (simulated) 
    audio_features = np.random.randn(n_samples, 32)
    
    # Video features (simulated)
    video_features = np.random.randn(n_samples, 48)
    
    # Combine all features
    X = np.hstack([image_features, audio_features, video_features])
    
    # Create labels with some correlation to features
    y = (X[:, 0] + X[:, 64] + X[:, 96] > 0).astype(int)
    
    return X, y

def train_all_models(X, y):
    """Train multiple models"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'SVM': SVC(kernel='rbf', random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"{name} Accuracy: {accuracy:.3f}")
        
        # Save model
        model_path = f'models/saved_models/{name.lower()}_model.pkl'
        joblib.dump(model, model_path)
        print(f"Saved to {model_path}")
        
        results[name] = accuracy
    
    return results

def main():
    print("=" * 60)
    print("DEEPFAKE DETECTION - TRAINING ALL MODELS")
    print("=" * 60)
    
    # Create dataset
    X, y = create_multimodal_dataset()
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Real: {np.sum(y == 0)}, Fake: {np.sum(y == 1)}")
    
    # Train models
    results = train_all_models(X, y)
    
    print("\n" + "=" * 60)
    print("TRAINING RESULTS:")
    print("=" * 60)
    for model, accuracy in results.items():
        print(f"{model:20}: {accuracy:.3f}")
    
    print(f"\nBest model: {max(results, key=results.get)} ({max(results.values()):.3f})")

if __name__ == "__main__":
    main()