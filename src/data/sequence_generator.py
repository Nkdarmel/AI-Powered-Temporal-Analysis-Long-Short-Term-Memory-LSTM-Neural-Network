"""Generate sequences for LSTM training"""

import logging
import numpy as np
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SequenceGenerator:
    """Generate sequences from time-series data for LSTM models"""
    
    def __init__(self, sequence_length: int = 50, stride: int = 1):
        """
        Initialize sequence generator
        
        Args:
            sequence_length: Length of each sequence (default: 50 timesteps)
            stride: Step size between sequences (default: 1)
        """
        self.sequence_length = sequence_length
        self.stride = stride
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences from data
        
        Args:
            X: Features array (n_samples, n_features)
            y: Labels array (n_samples,)
            
        Returns:
            Tuple of (sequences, labels) where sequences shape is (n_sequences, sequence_length, n_features)
        """
        sequences = []
        labels = []
        
        for i in range(0, len(X) - self.sequence_length, self.stride):
            seq = X[i:i + self.sequence_length]
            label = y[i + self.sequence_length - 1]  # Label of last sample in sequence
            
            sequences.append(seq)
            labels.append(label)
        
        sequences = np.array(sequences)
        labels = np.array(labels)
        
        logger.info(f"Created {len(sequences)} sequences of length {self.sequence_length}")
        logger.info(f"Sequences shape: {sequences.shape}")
        logger.info(f"Labels shape: {labels.shape}")
        
        return sequences, labels
    
    def pad_sequences(self, sequences: list, max_length: Optional[int] = None,
                      pad_value: float = 0.0) -> np.ndarray:
        """
        Pad sequences to uniform length
        
        Args:
            sequences: List of variable-length sequences
            max_length: Maximum length (if None, use max of input sequences)
            pad_value: Value to use for padding
            
        Returns:
            Padded sequences array (n_sequences, max_length, n_features)
        """
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
        
        padded = []
        for seq in sequences:
            if len(seq) < max_length:
                pad_width = ((0, max_length - len(seq)), (0, 0))
                padded_seq = np.pad(seq, pad_width, constant_values=pad_value)
            else:
                padded_seq = seq[:max_length]
            padded.append(padded_seq)
        
        padded = np.array(padded)
        logger.info(f"Padded sequences to shape: {padded.shape}")
        
        return padded