"""
Helper module for video and audio processing for deepfake detection.
Provides frame extraction, audio feature extraction, and analysis utilities.
"""

import os
import tempfile
import random
import logging

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import librosa
    import numpy as np
except ImportError:
    librosa = None
    np = None


def extract_video_frames(video_path, max_frames=10, sample_rate=None):
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract (evenly distributed)
        sample_rate: Sample every N frames (if None, evenly space max_frames)
    
    Returns:
        List of PIL Image objects or None if video processing failed
    """
    if cv2 is None:
        logger.warning("OpenCV (cv2) not installed; video processing disabled")
        return None
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames == 0:
            logger.error("Video has no frames")
            cap.release()
            return None
        
        # Calculate frame indices to sample
        if sample_rate is None:
            # Evenly space max_frames across the video
            frame_indices = [int(i * total_frames / max_frames) for i in range(max_frames)]
        else:
            frame_indices = list(range(0, total_frames, sample_rate))[:max_frames]
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count in frame_indices:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                try:
                    from PIL import Image
                    pil_image = Image.fromarray(frame_rgb)
                    frames.append(pil_image)
                except ImportError:
                    logger.warning("PIL not available; frame conversion failed")
                    break
            
            frame_count += 1
        
        cap.release()
        
        logger.info(f"Extracted {len(frames)} frames from video")
        return frames if frames else None
    
    except Exception as e:
        logger.error(f"Error extracting video frames: {e}")
        return None


def analyze_audio_features(audio_path):
    """
    Extract audio features for deepfake detection.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Dict with audio metadata and features, or None if analysis failed
    """
    if librosa is None or np is None:
        logger.warning("Librosa or NumPy not installed; audio feature extraction disabled")
        return None
    
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Extract basic features
        features = {
            'sample_rate': sr,
            'duration': duration,
            'rms_energy': float(np.mean(librosa.feature.rms(y=y))),
            'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(y))),
        }
        
        # Extract MFCC (Mel-Frequency Cepstral Coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = float(np.mean(mfcc, axis=1).mean())
        features['mfcc_std'] = float(np.std(mfcc, axis=1).mean())
        
        logger.info(f"Extracted audio features: {list(features.keys())}")
        return features
    
    except Exception as e:
        logger.error(f"Error extracting audio features: {e}")
        return None


def get_video_metadata(video_path):
    """
    Get basic metadata about a video file.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dict with metadata or None if failed
    """
    if cv2 is None:
        return None
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        metadata = {
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration_sec': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return metadata
    
    except Exception as e:
        logger.error(f"Error reading video metadata: {e}")
        return None


def dummy_video_prediction(num_frames):
    """
    Generate dummy predictions for video frames (for demo/fallback purposes).
    Returns predictions for each frame.
    """
    predictions = []
    for _ in range(num_frames):
        pred = {
            'label': random.choice(['Real', 'Fake']),
            'confidence': random.uniform(0.5, 0.99)
        }
        predictions.append(pred)
    return predictions


def dummy_audio_prediction():
    """
    Generate dummy prediction for audio (for demo/fallback purposes).
    """
    return {
        'label': random.choice(['Real', 'Fake']),
        'confidence': random.uniform(0.5, 0.99),
        'message': 'Dummy audio prediction (requires audio models)'
    }
