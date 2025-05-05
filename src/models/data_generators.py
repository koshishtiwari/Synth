import numpy as np
import pandas as pd
import random
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from faker import Faker

from src.core.data_generator import BaseDataGenerator
from src.config.config_manager import config_manager

class TabularDataGenerator(BaseDataGenerator):
    """Generator for tabular data with customizable schemas.
    
    Supports various data types including numeric, categorical, datetime, etc.
    Can apply controlled drift to the generated data.
    """
    
    def __init__(self, name: str, schema: Dict[str, Any]):
        """Initialize the tabular data generator.
        
        Args:
            name: Name of the generator.
            schema: Schema definition for the tabular data.
                    Should contain field definitions with types and parameters.
        """
        super().__init__(name, schema)
        self.faker = Faker()
        self.faker.seed_instance(self.seed)
        self.drift_params = {}
        
        # Initialize distribution parameters for each field
        for field_name, field_def in self.schema.items():
            field_type = field_def.get('type', 'float')
            
            if field_type == 'float':
                # Store mean and std for numeric fields
                self.drift_params[field_name] = {
                    'mean': field_def.get('mean', 0.0),
                    'std': field_def.get('std', 1.0),
                    'min': field_def.get('min', float('-inf')),
                    'max': field_def.get('max', float('inf')),
                }
            elif field_type == 'int':
                # Store min and max for integer fields
                self.drift_params[field_name] = {
                    'min': field_def.get('min', 0),
                    'max': field_def.get('max', 100),
                }
            elif field_type == 'category':
                # Store probabilities for categorical fields
                categories = field_def.get('categories', ['A', 'B', 'C'])
                probs = field_def.get('probabilities', None)
                if probs is None:
                    probs = [1.0/len(categories)] * len(categories)
                self.drift_params[field_name] = {
                    'categories': categories,
                    'probabilities': probs,
                }
    
    def _generate_float(self, field_name: str, size: int) -> np.ndarray:
        """Generate float values based on field parameters.
        
        Args:
            field_name: The name of the field to generate.
            size: Number of values to generate.
            
        Returns:
            Array of generated float values.
        """
        params = self.drift_params[field_name]
        values = self.rng.normal(params['mean'], params['std'], size)
        # Apply min/max bounds
        values = np.clip(values, params['min'], params['max'])
        return values
    
    def _generate_int(self, field_name: str, size: int) -> np.ndarray:
        """Generate integer values based on field parameters.
        
        Args:
            field_name: The name of the field to generate.
            size: Number of values to generate.
            
        Returns:
            Array of generated integer values.
        """
        params = self.drift_params[field_name]
        values = self.rng.randint(params['min'], params['max'] + 1, size)
        return values
    
    def _generate_category(self, field_name: str, size: int) -> List[str]:
        """Generate categorical values based on field parameters.
        
        Args:
            field_name: The name of the field to generate.
            size: Number of values to generate.
            
        Returns:
            List of generated categorical values.
        """
        params = self.drift_params[field_name]
        values = self.rng.choice(
            params['categories'], 
            size=size, 
            p=params['probabilities']
        )
        return values
    
    def _generate_datetime(self, field_def: Dict[str, Any], size: int) -> List[datetime]:
        """Generate datetime values based on field parameters.
        
        Args:
            field_def: The field definition.
            size: Number of values to generate.
            
        Returns:
            List of generated datetime values.
        """
        start_date = field_def.get('start_date', datetime(2020, 1, 1))
        end_date = field_def.get('end_date', datetime(2023, 12, 31))
        
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)
            
        start_ts = start_date.timestamp()
        end_ts = end_date.timestamp()
        
        # Generate random timestamps between start and end
        timestamps = self.rng.uniform(start_ts, end_ts, size)
        dates = [datetime.fromtimestamp(ts) for ts in timestamps]
        return dates
    
    def _generate_email(self, size: int) -> List[str]:
        """Generate email addresses.
        
        Args:
            size: Number of values to generate.
            
        Returns:
            List of generated email addresses.
        """
        return [self.faker.email() for _ in range(size)]
    
    def _generate_name(self, size: int) -> List[str]:
        """Generate person names.
        
        Args:
            size: Number of values to generate.
            
        Returns:
            List of generated names.
        """
        return [self.faker.name() for _ in range(size)]
    
    def _generate_address(self, size: int) -> List[str]:
        """Generate addresses.
        
        Args:
            size: Number of values to generate.
            
        Returns:
            List of generated addresses.
        """
        return [self.faker.address().replace('\n', ', ') for _ in range(size)]
    
    def generate_batch(self, size: int) -> pd.DataFrame:
        """Generate a batch of tabular data.
        
        Args:
            size: Number of records to generate.
            
        Returns:
            DataFrame with generated data.
        """
        data = {}
        
        for field_name, field_def in self.schema.items():
            field_type = field_def.get('type', 'float')
            
            if field_type == 'float':
                data[field_name] = self._generate_float(field_name, size)
            elif field_type == 'int':
                data[field_name] = self._generate_int(field_name, size)
            elif field_type == 'category':
                data[field_name] = self._generate_category(field_name, size)
            elif field_type == 'datetime':
                data[field_name] = self._generate_datetime(field_def, size)
            elif field_type == 'email':
                data[field_name] = self._generate_email(size)
            elif field_type == 'name':
                data[field_name] = self._generate_name(size)
            elif field_type == 'address':
                data[field_name] = self._generate_address(size)
            else:
                self.logger.warning(f"Unknown field type: {field_type} for field {field_name}")
                data[field_name] = [None] * size
                
        return pd.DataFrame(data)
    
    def apply_drift(self, data: pd.DataFrame, magnitude: float) -> pd.DataFrame:
        """Apply controlled drift to the generated data.
        
        Args:
            data: The data to apply drift to.
            magnitude: The magnitude of the drift (0.0 to 1.0).
            
        Returns:
            DataFrame with drift applied.
        """
        drifted_data = data.copy()
        
        for field_name, field_def in self.schema.items():
            field_type = field_def.get('type', 'float')
            
            if field_type == 'float':
                # Drift the mean for numeric fields
                params = self.drift_params[field_name]
                drift_direction = self.rng.choice([-1, 1])
                drift_amount = magnitude * params['std'] * drift_direction
                
                # Apply the drift
                drifted_data[field_name] = data[field_name] + drift_amount
                
                # Ensure bounds are respected
                drifted_data[field_name] = np.clip(
                    drifted_data[field_name], 
                    params['min'], 
                    params['max']
                )
                
            elif field_type == 'category':
                # Drift the probabilities for categorical fields
                if self.rng.random() < magnitude:
                    params = self.drift_params[field_name]
                    categories = params['categories']
                    orig_probs = params['probabilities']
                    
                    # Create drifted probabilities
                    drifted_probs = np.array(orig_probs) + self.rng.normal(0, magnitude, len(orig_probs))
                    drifted_probs = np.clip(drifted_probs, 0.01, 1.0)
                    drifted_probs = drifted_probs / drifted_probs.sum()  # Normalize
                    
                    # Apply drift by resampling some values
                    drift_mask = self.rng.random(len(data)) < magnitude
                    if drift_mask.any():
                        new_values = self.rng.choice(
                            categories,
                            size=drift_mask.sum(),
                            p=drifted_probs
                        )
                        drifted_data.loc[drift_mask, field_name] = new_values
        
        return drifted_data


class TimeSeriesGenerator(TabularDataGenerator):
    """Generator for time series data.
    
    Extends the tabular data generator with time series specific functionality.
    """
    
    def __init__(self, name: str, schema: Dict[str, Any]):
        """Initialize the time series generator.
        
        Args:
            name: Name of the generator.
            schema: Schema definition for the time series data.
        """
        super().__init__(name, schema)
        self.current_time = datetime.now()
        
        # Get time series specific settings
        self.time_field = self.schema.get('time_field', 'timestamp')
        self.time_interval = timedelta(seconds=self.schema.get('time_interval_seconds', 60))
        
        # Trends and seasonality
        self.trends = self.schema.get('trends', {})
        self.seasonality = self.schema.get('seasonality', {})
    
    def generate_batch(self, size: int) -> pd.DataFrame:
        """Generate a batch of time series data.
        
        Args:
            size: Number of records to generate.
            
        Returns:
            DataFrame with generated time series data.
        """
        # Generate base data
        data = super().generate_batch(size)
        
        # Add time component
        timestamps = [self.current_time + i * self.time_interval for i in range(size)]
        data[self.time_field] = timestamps
        
        # Apply time-based patterns
        for field_name, field_def in self.schema.items():
            if field_name == self.time_field or field_def.get('type') not in ['float', 'int']:
                continue
                
            # Apply trend if configured
            if field_name in self.trends:
                trend_type = self.trends[field_name].get('type', 'linear')
                trend_strength = self.trends[field_name].get('strength', 0.1)
                
                if trend_type == 'linear':
                    # Linear trend over time
                    trend = np.linspace(0, trend_strength, size)
                    data[field_name] = data[field_name] + trend
                elif trend_type == 'exponential':
                    # Exponential trend over time
                    trend = np.exp(np.linspace(0, trend_strength, size)) - 1
                    data[field_name] = data[field_name] * (1 + trend)
            
            # Apply seasonality if configured
            if field_name in self.seasonality:
                period = self.seasonality[field_name].get('period', 24)  # Default daily period (24 hours)
                amplitude = self.seasonality[field_name].get('amplitude', 0.1)
                
                # Generate seasonal pattern
                t = np.array([(self.current_time + i * self.time_interval).timestamp() for i in range(size)])
                season = amplitude * np.sin(2 * np.pi * t / (period * 3600))  # Convert period hours to seconds
                
                data[field_name] = data[field_name] + season
        
        # Update current time for next batch
        self.current_time = timestamps[-1] + self.time_interval
        
        return data