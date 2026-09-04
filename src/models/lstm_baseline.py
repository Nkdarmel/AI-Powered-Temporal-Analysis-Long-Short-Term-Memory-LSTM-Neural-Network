"""Baseline LSTM model for anomaly detection"""

import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMBaseline:
    """Standard LSTM model for network anomaly detection"""
    
    def __init__(self, input_shape: Tuple[int, int], lstm_units: list = None,
                 dropout_rate: float = 0.3, activation: str = 'relu'):
        """
        Initialize LSTM model
        
        Args:
            input_shape: Tuple of (sequence_length, n_features)
            lstm_units: List of LSTM units per layer (default: [64, 32])
            dropout_rate: Dropout rate between layers (default: 0.3)
            activation: Activation function (default: 'relu')
        """
        self.input_shape = input_shape
        self.lstm_units = lstm_units or [64, 32]
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.model = None
        self._build_model()
    
    def _build_model(self):
        """Build the LSTM model architecture"""
        logger.info(f"Building LSTM model with input shape: {self.input_shape}")
        
        model = keras.Sequential()
        
        # First LSTM layer
        model.add(layers.LSTM(self.lstm_units[0], 
                              return_sequences=True,
                              input_shape=self.input_shape))
        model.add(layers.Dropout(self.dropout_rate))
        
        # Additional LSTM layers
        for units in self.lstm_units[1:]:
            model.add(layers.LSTM(units, return_sequences=True))
            model.add(layers.Dropout(self.dropout_rate))
        
        # Dense layers
        model.add(layers.LSTM(16, return_sequences=False))
        model.add(layers.Dropout(self.dropout_rate))
        model.add(layers.Dense(64, activation=self.activation))
        model.add(layers.Dropout(self.dropout_rate))
        model.add(layers.Dense(32, activation=self.activation))
        model.add(layers.Dropout(self.dropout_rate))
        
        # Output layer (binary classification)
        model.add(layers.Dense(1, activation='sigmoid'))
        
        self.model = model
        logger.info("Model built successfully")
    
    def compile(self, optimizer: str = 'adam', loss: str = 'binary_crossentropy',
                metrics: list = None):
        """
        Compile the model
        
        Args:
            optimizer: Optimizer (default: 'adam')
            loss: Loss function (default: 'binary_crossentropy')
            metrics: Metrics to track (default: accuracy, precision, recall)
        """
        if metrics is None:
            metrics = ['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        logger.info(f"Model compiled with {optimizer} optimizer")
    
    def get_model(self) -> keras.Sequential:
        """Return the Keras model"""
        return self.model
    
    def summary(self):
        """Print model summary"""
        self.model.summary()