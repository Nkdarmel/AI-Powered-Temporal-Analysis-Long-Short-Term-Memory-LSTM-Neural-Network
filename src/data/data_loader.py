"""Data loading and preprocessing module"""

import os
import logging
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and prepare network traffic data for LSTM model training"""
    
    def __init__(self, data_path: str, test_size: float = 0.2, 
                 val_size: float = 0.1, random_state: int = 42):
        """
        Initialize DataLoader
        
        Args:
            data_path: Path to CSV file containing network traffic data
            test_size: Fraction of data for testing (default: 0.2)
            val_size: Fraction of training data for validation (default: 0.1)
            random_state: Random seed for reproducibility
        """
        self.data_path = data_path
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.scaler = MinMaxScaler()
        self.data = None
        self.feature_cols = None
        self.label_col = None
        
    def load_data(self) -> pd.DataFrame:
        """Load CSV data from file"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        logger.info(f"Loading data from {self.data_path}")
        self.data = pd.read_csv(self.data_path)
        logger.info(f"Data shape: {self.data.shape}")
        logger.info(f"Columns: {self.data.columns.tolist()}")
        
        return self.data
    
    def validate_data(self) -> bool:
        """Validate data quality"""
        if self.data is None:
            logger.error("Data not loaded")
            return False
        
        # Check for missing values
        missing = self.data.isnull().sum()
        if missing.any():
            logger.warning(f"Missing values detected:\n{missing[missing > 0]}")
        
        # Check for duplicate rows
        duplicates = self.data.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate rows")
        
        logger.info("Data validation completed")
        return True
    
    def load_and_split(self, feature_cols: Optional[list] = None, 
                       label_col: str = 'label') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data and split into train/val/test sets
        
        Args:
            feature_cols: List of feature column names (if None, use all except label)
            label_col: Name of label column
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Load data
        self.load_data()
        self.validate_data()
        
        # Set feature columns
        if feature_cols is None:
            self.feature_cols = [col for col in self.data.columns if col != label_col]
        else:
            self.feature_cols = feature_cols
        
        self.label_col = label_col
        
        # Separate features and labels
        X = self.data[self.feature_cols].values
        y = self.data[label_col].values
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Labels shape: {y.shape}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        logger.info(f"Train set: {X_train.shape}")
        logger.info(f"Test set: {X_test.shape}")
        logger.info(f"Class distribution - Train: {np.bincount(y_train.astype(int))}")
        logger.info(f"Class distribution - Test: {np.bincount(y_test.astype(int))}")
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_names(self) -> list:
        """Get feature column names"""
        return self.feature_cols
    
    def get_scaler(self):
        """Get the fitted scaler (useful for inference)"""
        return self.scaler