"""Model training utilities"""

import logging
import numpy as np
from typing import Tuple, Optional, Dict
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handle model training, validation, and evaluation"""
    
    def __init__(self, model):
        """
        Initialize trainer
        
        Args:
            model: Keras model to train
        """
        self.model = model
        self.history = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              epochs: int = 100, batch_size: int = 32,
              validation_split: float = 0.2,
              early_stopping: bool = True) -> Dict:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            epochs: Number of epochs
            batch_size: Batch size
            validation_split: Validation split fraction
            early_stopping: Whether to use early stopping
            
        Returns:
            Training history
        """
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Train data shape: {X_train.shape}")
        logger.info(f"Batch size: {batch_size}")
        
        callbacks = []
        
        if early_stopping:
            callbacks.append(EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ))
            callbacks.append(ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            ))
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training completed")
        return self.history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 batch_size: int = 32) -> Dict:
        """
        Evaluate model on test data
        
        Args:
            X_test: Test features
            y_test: Test labels
            batch_size: Batch size
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating on test data: {X_test.shape}")
        
        results = self.model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
        
        metrics_dict = {
            'loss': results[0],
            'accuracy': results[1] if len(results) > 1 else None,
            'precision': results[2] if len(results) > 2 else None,
            'recall': results[3] if len(results) > 3 else None,
        }
        
        logger.info(f"Test Metrics: {metrics_dict}")
        return metrics_dict
    
    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
            batch_size: Batch size
            
        Returns:
            Predictions array
        """
        predictions = self.model.predict(X, batch_size=batch_size, verbose=0)
        return predictions
    
    def save_model(self, path: str):
        """
        Save model to disk
        
        Args:
            path: Path to save the model
        """
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load model from disk
        
        Args:
            path: Path to load the model from
        """
        self.model = keras.models.load_model(path)
        logger.info(f"Model loaded from {path}")