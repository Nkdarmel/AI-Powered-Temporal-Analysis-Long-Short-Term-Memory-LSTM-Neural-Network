"""Data preprocessing utilities"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handle data normalization, scaling, and feature engineering"""
    
    def __init__(self, scaler_type: str = 'minmax'):
        """
        Initialize preprocessor
        
        Args:
            scaler_type: Type of scaler - 'minmax', 'standard', or 'robust'
        """
        self.scaler_type = scaler_type
        self.scaler = None
        self._init_scaler()
    
    def _init_scaler(self):
        """Initialize the appropriate scaler"""
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
        logger.info(f"Initialized {self.scaler_type} scaler")
    
    def normalize(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize features
        
        Args:
            X: Input features
            fit: Whether to fit the scaler
            
        Returns:
            Normalized features
        """
        if fit:
            X_normalized = self.scaler.fit_transform(X)
        else:
            X_normalized = self.scaler.transform(X)
        
        logger.info(f"Normalized data shape: {X_normalized.shape}")
        return X_normalized
    
    def remove_outliers(self, X: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """
        Remove outliers using z-score
        
        Args:
            X: Input features
            threshold: Z-score threshold (default: 3.0)
            
        Returns:
            Data with outliers removed
        """
        z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
        mask = (z_scores < threshold).all(axis=1)
        
        removed = (~mask).sum()
        logger.info(f"Removed {removed} outliers")
        
        return X[mask]
    
    def handle_missing_values(self, X: np.ndarray, method: str = 'mean') -> np.ndarray:
        """
        Handle missing values
        
        Args:
            X: Input features with NaN values
            method: 'mean', 'median', or 'forward_fill'
            
        Returns:
            Data with missing values handled
        """
        X_copy = X.copy()
        
        if method == 'mean':
            col_means = np.nanmean(X_copy, axis=0)
            mask = np.isnan(X_copy)
            X_copy[mask] = col_means[np.where(mask)[1]]
        elif method == 'median':
            col_medians = np.nanmedian(X_copy, axis=0)
            mask = np.isnan(X_copy)
            X_copy[mask] = col_medians[np.where(mask)[1]]
        
        logger.info(f"Handled missing values using {method}")
        return X_copy
    
    def get_scaler(self):
        """Return the fitted scaler"""
        return self.scaler