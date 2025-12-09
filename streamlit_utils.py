"""
Streamlit Utilities - Helper functions for model inference and visualization
"""

import torch
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Tuple, Dict, Any, Optional
import os

logger = logging.getLogger(__name__)

MODEL_DIR = "models/saved_models"


def load_cnn_model():
    """Load CNN model from checkpoint."""
    try:
        from src.models.cnn import CNNModel
        model_path = os.path.join(MODEL_DIR, "cnn_model.pth")
        if not os.path.exists(model_path):
            logger.warning(f"CNN model not found at {model_path}")
            return None
        
        model = CNNModel(num_classes=2, input_channels=1)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        logger.info("✓ CNN model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading CNN model: {e}")
        return None


def load_transformer_model():
    """Load Transformer model from checkpoint."""
    try:
        from src.models.transformer import TransformerModel
        model_path = os.path.join(MODEL_DIR, "transformer_model.pth")
        if not os.path.exists(model_path):
            logger.warning(f"Transformer model not found at {model_path}")
            return None
        
        model = TransformerModel(
            input_dim=10, model_dim=512, num_heads=8,
            num_layers=6, output_dim=2
        )
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        logger.info("✓ Transformer model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading Transformer model: {e}")
        return None


def load_svm_model():
    """Load SVM model from pickle."""
    try:
        model_path = os.path.join(MODEL_DIR, "svm_model.pkl")
        if not os.path.exists(model_path):
            logger.warning(f"SVM model not found at {model_path}")
            return None
        
        model = joblib.load(model_path)
        logger.info("✓ SVM model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading SVM model: {e}")
        return None


def load_bayesian_model():
    """Load Bayesian model from pickle."""
    try:
        model_path = os.path.join(MODEL_DIR, "bayesian_model.pkl")
        if not os.path.exists(model_path):
            logger.warning(f"Bayesian model not found at {model_path}")
            return None
        
        model = joblib.load(model_path)
        logger.info("✓ Bayesian model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading Bayesian model: {e}")
        return None


def load_vision_transformer_model():
    """Load Vision Transformer model from checkpoint."""
    try:
        from src.models.vision_transformer import VisionTransformer
        model_path = os.path.join(MODEL_DIR, "vision_transformer_model.pth")
        if not os.path.exists(model_path):
            logger.warning(f"Vision Transformer model not found at {model_path}")
            return None
        
        model = VisionTransformer(
            img_size=8,
            patch_size=2,
            num_classes=2,
            dim=64,
            depth=2,
            heads=4,
            mlp_dim=128
        )
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        logger.info("✓ Vision Transformer model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading Vision Transformer model: {e}")
        return None


def load_all_models() -> Dict[str, Any]:
    """Load all available models."""
    models = {}
    
    cnn = load_cnn_model()
    if cnn:
        models['CNN'] = cnn
    
    transformer = load_transformer_model()
    if transformer:
        models['Transformer'] = transformer
    
    svm = load_svm_model()
    if svm:
        models['SVM'] = svm
    
    bayesian = load_bayesian_model()
    if bayesian:
        models['Bayesian'] = bayesian
    
    vit = load_vision_transformer_model()
    if vit:
        models['Vision Transformer'] = vit
    
    logger.info(f"Loaded {len(models)} models: {list(models.keys())}")
    return models


def preprocess_image(
    image_array: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    num_channels: int = 3,
) -> np.ndarray:
    """
    Preprocess image for CNN/ViT input.
    - Converts to float32 in [0,1]
    - Ensures `num_channels` channels by repeating or trimming
    - Resizes to `target_size`
    """
    try:
        img = np.asarray(image_array)

        # Ensure channel dimension present and has expected count
        if img.ndim == 2:
            img = np.repeat(img[:, :, np.newaxis], num_channels, axis=2)
        elif img.shape[2] < num_channels:
            img = np.repeat(img, num_channels // img.shape[2] + 1, axis=2)[:, :, :num_channels]
        else:
            img = img[:, :, :num_channels]

        # Resize
        from scipy.ndimage import zoom
        resize_factor = (
            target_size[0] / img.shape[0],
            target_size[1] / img.shape[1],
            1,
        )
        resized = zoom(img, resize_factor, order=1)

        # Normalize to 0-1
        normalized = resized.astype(np.float32) / 255.0
        return normalized
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return np.random.rand(target_size[0], target_size[1], num_channels).astype(
            np.float32
        )


def predict_cnn(model: Any, image_array: np.ndarray) -> Tuple[str, float]:
    """Run CNN prediction on image (returns label, confidence of predicted class)."""
    try:
        if model is None:
            return "Unknown", 0.5
        
        # Preprocess
        processed = preprocess_image(image_array, target_size=(224, 224), num_channels=3)
        
        # Convert to tensor
        tensor = torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            output = model(tensor)
            probabilities = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, pred_class].item()
        label = 'Real' if pred_class == 0 else 'Fake'
        return label, confidence
    except Exception as e:
        logger.error(f"Error in CNN prediction: {e}")
        return 'Unknown', 0.5


def predict_cnn_with_probs(model: Any, image_array: np.ndarray) -> Tuple[str, float, float]:
    """
    CNN prediction returning:
    - label (Real/Fake based on fake probability >= 0.5)
    - fake_prob (probability of class 1)
    - confidence (max class probability)
    """
    try:
        if model is None:
            return "Unknown", 0.5, 0.5

        processed = preprocess_image(image_array, target_size=(224, 224), num_channels=3)
        tensor = torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)
            probabilities = torch.softmax(output, dim=1)
            fake_prob = probabilities[0, 1].item()
            confidence = float(torch.max(probabilities).item())

        label = 'Fake' if fake_prob >= 0.5 else 'Real'
        return label, float(fake_prob), confidence
    except Exception as e:
        logger.error(f"Error in CNN prediction: {e}")
        return 'Unknown', 0.5, 0.5


def predict_transformer(model: Any, features: np.ndarray) -> Tuple[str, float]:
    """Run Transformer prediction on features."""
    try:
        if model is None:
            return "Unknown", 0.5
        
        # Ensure we have proper features
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Pad or truncate to 10 features
        if features.shape[1] < 10:
            features = np.pad(features, ((0, 0), (0, 10 - features.shape[1])), mode='constant')
        else:
            features = features[:, :10]
        
        # Convert to tensor
        tensor = torch.FloatTensor(features)
        
        # Predict
        with torch.no_grad():
            output = model(tensor)
            probabilities = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, pred_class].item()
        
        label = 'Real' if pred_class == 0 else 'Fake'
        return label, confidence
    except Exception as e:
        logger.error(f"Error in Transformer prediction: {e}")
        return 'Unknown', 0.5


def predict_svm(model: Any, features: np.ndarray) -> Tuple[str, float]:
    """Run SVM prediction on features."""
    try:
        if model is None:
            return "Unknown", 0.5
        
        # Ensure proper shape
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Pad or truncate to match training features
        # SVM expects consistent input size
        if features.shape[1] < 10:
            features = np.pad(features, ((0, 0), (0, 10 - features.shape[1])), mode='constant')
        else:
            features = features[:, :10]
        
        # Predict
        pred_class = model.predict(features)[0]
        
        # Get confidence if decision_function available
        try:
            decision = model.decision_function(features)[0]
            confidence = 1.0 / (1.0 + np.exp(-decision))  # Sigmoid
        except:
            confidence = 0.7
        
        label = 'Real' if pred_class == 0 else 'Fake'
        return label, confidence
    except Exception as e:
        logger.error(f"Error in SVM prediction: {e}")
        return 'Unknown', 0.5


def predict_bayesian(model: Any, features: np.ndarray) -> Tuple[str, float]:
    """Run Bayesian prediction on features."""
    try:
        if model is None:
            return "Unknown", 0.5
        
        # Ensure proper shape
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Pad or truncate
        if features.shape[1] < 10:
            features = np.pad(features, ((0, 0), (0, 10 - features.shape[1])), mode='constant')
        else:
            features = features[:, :10]
        
        # Predict with probability
        pred_class = model.predict(features)[0]
        pred_proba = model.predict_proba(features)[0]
        confidence = max(pred_proba)
        
        label = 'Real' if pred_class == 0 else 'Fake'
        return label, confidence
    except Exception as e:
        logger.error(f"Error in Bayesian prediction: {e}")
        return 'Unknown', 0.5


def predict_vision_transformer(model: Any, image_array: np.ndarray) -> Tuple[str, float]:
    """Run Vision Transformer prediction on image."""
    try:
        if model is None:
            return "Unknown", 0.5
        
        # Preprocess to 3-channel 8x8
        if len(image_array.shape) == 3:
            # Use only first 3 channels or replicate if needed
            if image_array.shape[2] >= 3:
                rgb = image_array[:, :, :3]
            else:
                rgb = np.repeat(image_array[:, :, :1], 3, axis=2)
        else:
            rgb = np.repeat(image_array[:, :, np.newaxis], 3, axis=2)
        
        # Resize to 8x8
        from scipy.ndimage import zoom
        resize_factor = (8 / rgb.shape[0], 8 / rgb.shape[1], 1)
        resized = zoom(rgb, resize_factor, order=1)
        
        # Normalize and convert to tensor
        normalized = (resized - resized.min()) / (resized.max() - resized.min() + 1e-8)
        tensor = torch.FloatTensor(normalized).permute(2, 0, 1).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            output = model(tensor)
            probabilities = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, pred_class].item()
        
        label = 'Real' if pred_class == 0 else 'Fake'
        return label, confidence
    except Exception as e:
        logger.error(f"Error in Vision Transformer prediction: {e}")
        return 'Unknown', 0.5


def ensemble_predict(models: Dict[str, Any], features: np.ndarray, image_array: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Run ensemble prediction using all available models."""
    predictions = {}
    
    if 'CNN' in models and image_array is not None:
        predictions['CNN'] = predict_cnn(models['CNN'], image_array)
    
    if 'Transformer' in models:
        predictions['Transformer'] = predict_transformer(models['Transformer'], features)
    
    if 'SVM' in models:
        predictions['SVM'] = predict_svm(models['SVM'], features)
    
    if 'Bayesian' in models:
        predictions['Bayesian'] = predict_bayesian(models['Bayesian'], features)
    
    if 'Vision Transformer' in models and image_array is not None:
        predictions['Vision Transformer'] = predict_vision_transformer(models['Vision Transformer'], image_array)
    
    # Calculate ensemble vote
    votes = [1 if pred[0] == 'Fake' else 0 for pred in predictions.values()]
    ensemble_label = 'Fake' if np.mean(votes) > 0.5 else 'Real'
    ensemble_confidence = np.mean([pred[1] for pred in predictions.values()])
    
    return {
        'individual_predictions': predictions,
        'ensemble_label': ensemble_label,
        'ensemble_confidence': ensemble_confidence
    }


def load_processed_data(data_type: str = 'train') -> Optional[pd.DataFrame]:
    """Load processed data from CSV."""
    try:
        if data_type == 'train':
            path = "data/processed/processed_train.csv"
        elif data_type == 'test':
            path = "data/processed/processed_test.csv"
        else:
            path = "data/processed/processed_data.csv"
        
        if not os.path.exists(path):
            logger.warning(f"Data file not found: {path}")
            return None
        
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} samples from {path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None


def get_model_statistics() -> Dict[str, Any]:
    """Get statistics about loaded models and data."""
    stats = {
        'model_files': {},
        'data_files': {},
        'timestamp': pd.Timestamp.now()
    }
    
    # Check model files
    if os.path.exists(MODEL_DIR):
        for file in os.listdir(MODEL_DIR):
            path = os.path.join(MODEL_DIR, file)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            stats['model_files'][file] = {
                'size_mb': round(size_mb, 2),
                'modified': pd.Timestamp(os.path.getmtime(path))
            }
    
    # Check data files
    data_dirs = ['data/processed', 'data/raw']
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith('.csv'):
                    path = os.path.join(data_dir, file)
                    size_kb = os.path.getsize(path) / 1024
                    key = f"{data_dir}/{file}"
                    stats['data_files'][key] = {
                        'size_kb': round(size_kb, 2),
                        'modified': pd.Timestamp(os.path.getmtime(path))
                    }
    
    return stats
