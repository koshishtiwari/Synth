import logging
import os
import joblib
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta

from src.config.config_manager import config_manager

class BaseMLModel(ABC):
    """Base class for all ML models used in synthetic data generation.
    
    This abstract class defines the interface that all ML models must implement
    to be compatible with the Synth system. It supports training on historical data,
    predicting from new data, and adapting to changing data distributions.
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """Initialize the ML model.
        
        Args:
            name: Name of the model.
            config: Configuration parameters for the model.
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.is_trained = False
        self.last_trained = None
        self.metadata = {}
        
    @abstractmethod
    def train(self, data: pd.DataFrame, target_columns: List[str], **kwargs) -> None:
        """Train the model on historical data.
        
        Args:
            data: Training data.
            target_columns: Columns to predict.
            **kwargs: Additional training parameters.
        """
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate predictions based on input data.
        
        Args:
            data: Input data for predictions.
            **kwargs: Additional prediction parameters.
        
        Returns:
            DataFrame with predictions.
        """
        pass
    
    @abstractmethod
    def generate(self, context: Optional[pd.DataFrame] = None, n_samples: int = 1, **kwargs) -> pd.DataFrame:
        """Generate synthetic data based on learned patterns.
        
        Args:
            context: Context data to condition the generation (optional).
            n_samples: Number of samples to generate.
            **kwargs: Additional generation parameters.
        
        Returns:
            DataFrame with generated synthetic data.
        """
        pass
    
    def update(self, new_data: pd.DataFrame, **kwargs) -> None:
        """Update the model with new data (online learning).
        
        Args:
            new_data: New data for updating the model.
            **kwargs: Additional update parameters.
        """
        # Default implementation is to retrain on new data
        if not kwargs.get('target_columns'):
            self.logger.warning("No target columns specified for update, using existing ones")
        
        target_columns = kwargs.get('target_columns', self.metadata.get('target_columns', []))
        if not target_columns:
            self.logger.error("No target columns available for update")
            return
            
        self.train(new_data, target_columns, **kwargs)
    
    def save(self, path: str) -> None:
        """Save the model to disk.
        
        Args:
            path: Path to save the model.
        """
        model_dir = os.path.dirname(path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        model_data = {
            'model': self,
            'metadata': {
                'name': self.name,
                'is_trained': self.is_trained,
                'last_trained': self.last_trained,
                'metadata': self.metadata
            }
        }
        
        try:
            joblib.dump(model_data, path)
            self.logger.info(f"Model saved to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save model to {path}: {e}")
            raise
    
    @classmethod
    def load(cls, path: str) -> 'BaseMLModel':
        """Load a model from disk.
        
        Args:
            path: Path to load the model from.
            
        Returns:
            Loaded model.
        """
        try:
            model_data = joblib.load(path)
            model = model_data['model']
            model.logger.info(f"Model loaded from {path}")
            return model
        except Exception as e:
            logging.error(f"Failed to load model from {path}: {e}")
            raise
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the model.
        
        Returns:
            Dictionary with model information.
        """
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'is_trained': self.is_trained,
            'last_trained': self.last_trained,
            'metadata': self.metadata
        }