import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

from src.config.config_manager import config_manager

class BaseDataGenerator(ABC):
    """Base class for all data generators.
    
    Provides common functionality and interface for generating synthetic data.
    """
    
    def __init__(self, name: str, schema: Dict[str, Any]):
        """Initialize the data generator.
        
        Args:
            name: Name of the generator.
            schema: Schema definition for the data to be generated.
        """
        self.name = name
        self.schema = schema
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.seed = config_manager.get('data_generation.seed', 42)
        self.rng = np.random.RandomState(self.seed)
        self.logger.info(f"Initialized {name} generator with seed {self.seed}")
        
    def set_seed(self, seed: int) -> None:
        """Set the random seed for the generator.
        
        Args:
            seed: The random seed.
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.logger.info(f"Set seed to {seed}")
        
    @abstractmethod
    def generate_batch(self, size: int) -> pd.DataFrame:
        """Generate a batch of synthetic data.
        
        Args:
            size: Number of records to generate.
            
        Returns:
            DataFrame containing the generated data.
        """
        pass
    
    def apply_drift(self, data: pd.DataFrame, magnitude: float) -> pd.DataFrame:
        """Apply drift to the generated data.
        
        Args:
            data: The generated data.
            magnitude: The magnitude of the drift to apply.
            
        Returns:
            DataFrame with drift applied.
        """
        # Default implementation does nothing
        return data
    
    def metadata(self) -> Dict[str, Any]:
        """Get metadata about the generator.
        
        Returns:
            Dictionary with metadata.
        """
        return {
            "name": self.name,
            "schema": self.schema,
            "seed": self.seed,
        }