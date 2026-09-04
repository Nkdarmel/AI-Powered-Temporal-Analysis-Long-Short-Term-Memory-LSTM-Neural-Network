"""Model evaluation metrics"""

import logging
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Calculate evaluation metrics for model performance"""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                          threshold: float = 0.5) -> Dict:
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities or labels
            threshold: Classification threshold for probability predictions
            
        Returns:
            Dictionary of metrics
        """
        # Convert probabilities to binary predictions if needed
        if y_pred.max() <= 1.0 and y_pred.min() >= 0.0 and len(np.unique(y_pred)) > 2:
            y_pred_binary = (y_pred > threshold).astype(int).flatten()
        else:
            y_pred_binary = y_pred.flatten()
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred_binary),
            'precision': precision_score(y_true, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true, y_pred_binary, zero_division=0),
            'f1_score': f1_score(y_true, y_pred_binary, zero_division=0),
        }
        
        # Calculate ROC-AUC if predictions are probabilities
        if y_pred.max() <= 1.0 and y_pred.min() >= 0.0:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred.flatten())
            except:
                metrics['roc_auc'] = None
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_binary)
        metrics['confusion_matrix'] = cm
        metrics['tn'] = cm[0, 0]
        metrics['fp'] = cm[0, 1]
        metrics['fn'] = cm[1, 0]
        metrics['tp'] = cm[1, 1]
        
        # Additional metrics
        metrics['specificity'] = metrics['tn'] / (metrics['tn'] + metrics['fp']) if (metrics['tn'] + metrics['fp']) > 0 else 0
        metrics['sensitivity'] = metrics['tp'] / (metrics['tp'] + metrics['fn']) if (metrics['tp'] + metrics['fn']) > 0 else 0
        metrics['false_positive_rate'] = metrics['fp'] / (metrics['fp'] + metrics['tn']) if (metrics['fp'] + metrics['tn']) > 0 else 0
        
        logger.info(f"Metrics calculated - F1: {metrics['f1_score']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict):
        """
        Print metrics in formatted manner
        
        Args:
            metrics: Dictionary of metrics
        """
        print("\n" + "="*50)
        print("EVALUATION METRICS")
        print("="*50)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        if metrics.get('roc_auc'):
            print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"Specificity: {metrics.get('specificity', 0):.4f}")
        print(f"Sensitivity: {metrics.get('sensitivity', 0):.4f}")
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        print("="*50 + "\n")