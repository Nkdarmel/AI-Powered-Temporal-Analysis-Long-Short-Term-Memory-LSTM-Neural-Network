"""Real-time inference engine for predictions"""

import logging
import pickle
import numpy as np
from typing import Optional
from tensorflow import keras

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceEngine:
    """Perform real-time anomaly detection predictions"""
    
    def __init__(self, model_path: str, scaler_path: Optional[str] = None):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to saved model
            scaler_path: Path to saved scaler
        """
        self.model = keras.models.load_model(model_path)
        self.scaler = None
        
        if scaler_path:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        logger.info(f"InferenceEngine initialized with model: {model_path}")
    
    def preprocess(self, X: np.ndarray) -> np.ndarray:
        """
        Preprocess input for inference
        
        Args:
            X: Raw input features
            
        Returns:
            Preprocessed features
        """
        if self.scaler:
            X_processed = self.scaler.transform(X)
        else:
            X_processed = X
        
        return X_processed
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Make predictions on input data
        
        Args:
            X: Input features
            threshold: Classification threshold
            
        Returns:
            Binary predictions (0 for normal, 1 for anomaly)
        """
        # Preprocess
        X_processed = self.preprocess(X)
        
        # Get probabilities
        probabilities = self.model.predict(X_processed, verbose=0)
        
        # Apply threshold
        predictions = (probabilities > threshold).astype(int).flatten()
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            X: Input features
            
        Returns:
            Probability predictions
        """
        X_processed = self.preprocess(X)
        probabilities = self.model.predict(X_processed, verbose=0)
        return probabilities