import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.ml.base_model import BaseMLModel

class TimeSeriesModel(BaseMLModel):
    """Time series model for forecasting and generating synthetic time series data.
    
    Uses statistical time series models (ARIMA/SARIMA) for univariate series
    and machine learning models for multivariate time series.
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """Initialize the time series model.
        
        Args:
            name: Name of the model.
            config: Configuration parameters for the model.
                - model_type: Type of model ('arima', 'sarima', or 'ml').
                - order: Order of ARIMA model (p, d, q).
                - seasonal_order: Order of seasonal component for SARIMA (P, D, Q, s).
                - use_exog: Whether to use exogenous variables.
                - forecast_horizon: How many steps ahead to forecast.
                - historical_window: Size of historical window to use for context.
                - frequency: Data frequency (e.g., 'D' for daily, 'H' for hourly).
        """
        super().__init__(name, config)
        
        self.config.setdefault('model_type', 'arima')
        self.config.setdefault('order', (1, 1, 1))
        self.config.setdefault('seasonal_order', (1, 1, 1, 12))
        self.config.setdefault('use_exog', False)
        self.config.setdefault('forecast_horizon', 24)
        self.config.setdefault('historical_window', 72)
        self.config.setdefault('frequency', 'H')
        
        self.models = {}  # For storing models for each target column
        self.scalers = {}  # For storing scalers for each target column
        self.last_values = {}  # For storing last values for continuity
        self.preprocessor = None  # For ML feature preprocessing
    
    def train(self, data: pd.DataFrame, target_columns: List[str], **kwargs) -> None:
        """Train the time series model on historical data.
        
        Args:
            data: Training data with datetime index or timestamp column.
            target_columns: Columns to predict.
            **kwargs: Additional training parameters.
                - feature_columns: Columns to use as features (for ML models).
                - date_column: Name of datetime column if not index.
        """
        # Ensure data has datetime index
        date_column = kwargs.get('date_column', None)
        if date_column and date_column in data.columns:
            data = data.set_index(date_column)
        
        if not pd.api.types.is_datetime64_any_dtype(data.index):
            try:
                data.index = pd.to_datetime(data.index)
            except:
                self.logger.error("Failed to convert index to datetime. Ensure data has a proper datetime index.")
                return
        
        # Sort by datetime index
        data = data.sort_index()
        
        # Store target columns in metadata
        self.metadata['target_columns'] = target_columns
        
        # Store last values for each target
        for col in target_columns:
            if col in data.columns:
                self.last_values[col] = data[col].iloc[-self.config['historical_window']:]
        
        # Train a model for each target column
        model_type = self.config['model_type'].lower()
        feature_columns = kwargs.get('feature_columns', None)
        
        if model_type in ['arima', 'sarima']:
            # Train statistical models (univariate)
            for col in target_columns:
                if col not in data.columns:
                    self.logger.warning(f"Column {col} not found in data")
                    continue
                    
                target_data = data[col].astype(float)
                
                # Handle missing values
                target_data = target_data.interpolate(method='linear')
                
                try:
                    if model_type == 'arima':
                        model = ARIMA(
                            target_data,
                            order=self.config['order']
                        )
                        self.models[col] = model.fit()
                    else:  # sarima
                        model = SARIMAX(
                            target_data,
                            order=self.config['order'],
                            seasonal_order=self.config['seasonal_order']
                        )
                        self.models[col] = model.fit(disp=False)
                        
                    self.logger.info(f"Trained {model_type.upper()} model for {col}")
                except Exception as e:
                    self.logger.error(f"Failed to train {model_type.upper()} model for {col}: {e}")
                    continue
        
        elif model_type == 'ml':
            # Train ML model (multivariate)
            if not feature_columns:
                self.logger.warning("No feature columns specified for ML model. Using all columns except targets.")
                feature_columns = [col for col in data.columns if col not in target_columns]
            
            # Prepare feature preprocessing
            numeric_features = [col for col in feature_columns 
                               if pd.api.types.is_numeric_dtype(data[col])]
            categorical_features = [col for col in feature_columns 
                                   if pd.api.types.is_categorical_dtype(data[col]) or 
                                   pd.api.types.is_object_dtype(data[col])]
            
            preprocessor_steps = []
            if numeric_features:
                preprocessor_steps.append(('num', StandardScaler(), numeric_features))
            if categorical_features:
                preprocessor_steps.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features))
            
            if preprocessor_steps:
                self.preprocessor = ColumnTransformer(preprocessor_steps)
                self.preprocessor.fit(data[feature_columns])
            
            # Train a model for each target
            for col in target_columns:
                if col not in data.columns:
                    self.logger.warning(f"Column {col} not found in data")
                    continue
                
                # Prepare ML pipeline
                if self.preprocessor:
                    X = self.preprocessor.transform(data[feature_columns])
                else:
                    X = data[feature_columns].values
                    
                y = data[col].astype(float).values
                
                # Create and train the model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                try:
                    model.fit(X, y)
                    self.models[col] = model
                    
                    # Create a scaler for the target
                    scaler = StandardScaler()
                    scaler.fit(y.reshape(-1, 1))
                    self.scalers[col] = scaler
                    
                    self.logger.info(f"Trained ML model for {col}")
                except Exception as e:
                    self.logger.error(f"Failed to train ML model for {col}: {e}")
                    continue
        
        else:
            self.logger.error(f"Unsupported model type: {model_type}")
            return
        
        self.is_trained = True
        self.last_trained = datetime.now()
        self.logger.info(f"Trained {model_type.upper()} model on {len(data)} records")
    
    def predict(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate predictions based on input data.
        
        Args:
            data: Input data for predictions.
            **kwargs: Additional parameters.
                - steps: Number of steps to forecast (for time series models).
                - feature_columns: Feature columns for ML models.
                - date_column: Name of datetime column if not index.
        
        Returns:
            DataFrame with predictions.
        """
        if not self.is_trained:
            self.logger.error("Model is not trained yet")
            return pd.DataFrame()
        
        # Get parameters
        steps = kwargs.get('steps', self.config['forecast_horizon'])
        target_columns = self.metadata.get('target_columns', [])
        feature_columns = kwargs.get('feature_columns', None)
        date_column = kwargs.get('date_column', None)
        
        # Ensure data has datetime index for time series models
        if date_column and date_column in data.columns:
            data = data.set_index(date_column)
        
        if not pd.api.types.is_datetime64_any_dtype(data.index) and self.config['model_type'] in ['arima', 'sarima']:
            try:
                data.index = pd.to_datetime(data.index)
            except:
                self.logger.error("Failed to convert index to datetime. Ensure data has a proper datetime index.")
                return pd.DataFrame()
        
        results = {}
        
        model_type = self.config['model_type'].lower()
        if model_type in ['arima', 'sarima']:
            # Generate forecast for each target
            for col in target_columns:
                if col not in self.models:
                    self.logger.warning(f"No model available for {col}")
                    continue
                
                # For ARIMA/SARIMA, we need the most recent data to forecast
                model = self.models[col]
                
                try:
                    forecast = model.forecast(steps=steps)
                    results[col] = forecast.values
                except Exception as e:
                    self.logger.error(f"Failed to generate forecast for {col}: {e}")
                    continue
        
        elif model_type == 'ml':
            # Use ML model for predictions
            if not feature_columns and self.preprocessor:
                feature_columns = self.preprocessor.get_feature_names_out()
            
            if not feature_columns:
                self.logger.error("No feature columns available for prediction")
                return pd.DataFrame()
            
            # Transform features
            if self.preprocessor:
                X = self.preprocessor.transform(data[feature_columns])
            else:
                X = data[feature_columns].values
            
            # Generate predictions for each target
            for col in target_columns:
                if col not in self.models:
                    self.logger.warning(f"No model available for {col}")
                    continue
                
                model = self.models[col]
                
                try:
                    predictions = model.predict(X)
                    
                    # Inverse transform if scaler exists
                    if col in self.scalers:
                        predictions = self.scalers[col].inverse_transform(predictions.reshape(-1, 1)).flatten()
                    
                    results[col] = predictions
                except Exception as e:
                    self.logger.error(f"Failed to generate predictions for {col}: {e}")
                    continue
        
        # Create result DataFrame
        if not results:
            self.logger.warning("No predictions generated")
            return pd.DataFrame()
        
        if model_type in ['arima', 'sarima']:
            # For time series forecasts, create future dates
            last_date = data.index[-1] if not data.empty else datetime.now()
            if isinstance(last_date, pd.Timestamp):
                last_date = last_date.to_pydatetime()
            
            freq = self.config['frequency']
            future_dates = pd.date_range(
                start=last_date + timedelta(hours=1),  # Start from next hour
                periods=steps,
                freq=freq
            )
            
            result_df = pd.DataFrame(index=future_dates)
            for col, values in results.items():
                result_df[col] = values
        else:
            # For ML predictions, use the same index as input data
            result_df = pd.DataFrame(index=data.index)
            for col, values in results.items():
                result_df[col] = values
        
        self.logger.info(f"Generated predictions for {len(result_df)} time points")
        return result_df
    
    def generate(self, context: Optional[pd.DataFrame] = None, n_samples: int = 1, **kwargs) -> pd.DataFrame:
        """Generate synthetic time series data based on learned patterns.
        
        Args:
            context: Context data to condition the generation (optional).
            n_samples: Number of time steps to generate.
            **kwargs: Additional generation parameters.
                - noise_level: Amount of noise to add (0.0 to 1.0).
                - start_date: Start date for generated time series.
        
        Returns:
            DataFrame with generated synthetic data.
        """
        if not self.is_trained:
            self.logger.error("Model is not trained yet")
            return pd.DataFrame()
        
        # Get parameters
        noise_level = kwargs.get('noise_level', 0.1)
        start_date = kwargs.get('start_date', datetime.now())
        target_columns = self.metadata.get('target_columns', [])
        
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        
        # Generate future dates
        freq = self.config['frequency']
        future_dates = pd.date_range(
            start=start_date,
            periods=n_samples,
            freq=freq
        )
        
        # Initialize result DataFrame
        result_df = pd.DataFrame(index=future_dates)
        
        model_type = self.config['model_type'].lower()
        if model_type in ['arima', 'sarima']:
            # Generate data for each target
            for col in target_columns:
                if col not in self.models:
                    self.logger.warning(f"No model available for {col}")
                    continue
                
                model = self.models[col]
                
                try:
                    # Generate forecast
                    forecast = model.forecast(steps=n_samples)
                    values = forecast.values
                    
                    # Add noise
                    if noise_level > 0:
                        noise = np.random.normal(0, noise_level * np.std(values), size=len(values))
                        values = values + noise
                    
                    result_df[col] = values
                except Exception as e:
                    self.logger.error(f"Failed to generate synthetic data for {col}: {e}")
                    continue
        
        elif model_type == 'ml':
            # For ML models, we need to generate feature data first
            # This is complex and depends on the specific use case
            # For this example, we'll use a simplified approach:
            # 1. Start with the last known values (or zeros)
            # 2. Generate new points using the model
            # 3. Feed each point back as input to generate the next point
            
            if context is not None and not context.empty:
                # Use provided context as starting point
                current_data = context.copy()
            else:
                # Create empty starting data
                self.logger.warning("No context provided, starting with zeros")
                # Create empty DataFrame with necessary columns
                if self.preprocessor:
                    feature_names = self.preprocessor.get_feature_names_out()
                    current_data = pd.DataFrame(0, index=[0], columns=feature_names)
                else:
                    self.logger.error("No preprocessor available and no context provided")
                    return pd.DataFrame()
            
            # Generate one point at a time, feeding each back as input
            all_generated = []
            
            for i in range(n_samples):
                # Prepare features
                if self.preprocessor:
                    X = self.preprocessor.transform(current_data)
                else:
                    X = current_data.values
                
                # Generate predictions for each target
                current_predictions = {}
                for col in target_columns:
                    if col not in self.models:
                        continue
                    
                    model = self.models[col]
                    pred = model.predict(X)[-1]  # Take the last prediction
                    
                    # Inverse transform if scaler exists
                    if col in self.scalers:
                        pred = self.scalers[col].inverse_transform([[pred]])[0, 0]
                        
                    # Add noise
                    if noise_level > 0:
                        noise = np.random.normal(0, noise_level * abs(pred) * 0.1)
                        pred = pred + noise
                        
                    current_predictions[col] = pred
                
                # Store the generated point
                current_time = future_dates[i]
                point_df = pd.DataFrame(current_predictions, index=[current_time])
                all_generated.append(point_df)
                
                # Use this prediction as input for the next step
                # This part would need to be customized based on the specific model and features
                # For this example, we'll just copy the last row for simplicity
                current_data = current_data.copy()
                
            # Combine all generated points
            if all_generated:
                result_df = pd.concat(all_generated)
            
        self.logger.info(f"Generated {len(result_df)} synthetic time points")
        return result_df

class GANTimeSeriesModel(BaseMLModel):
    """GAN-based time series model for generating highly realistic synthetic data.
    
    Uses Generative Adversarial Networks to learn the distribution of time series data
    and generate new samples.
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """Initialize the GAN time series model.
        
        Args:
            name: Name of the model.
            config: Configuration parameters for the model.
        """
        super().__init__(name, config)
        self.logger.info("GAN model initialized - placeholder implementation")
        
    def train(self, data: pd.DataFrame, target_columns: List[str], **kwargs) -> None:
        """Train the GAN model on historical data.
        
        This is a placeholder implementation. In a real-world scenario,
        this would involve training a GAN model on the provided data.
        
        Args:
            data: Training data.
            target_columns: Columns to generate.
            **kwargs: Additional training parameters.
        """
        self.logger.info(f"Training GAN model (placeholder) on {len(data)} records")
        self.is_trained = True
        self.last_trained = datetime.now()
        self.metadata['target_columns'] = target_columns
        
    def predict(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate predictions using the GAN model.
        
        Placeholder implementation.
        
        Args:
            data: Input data for conditioning the predictions.
            **kwargs: Additional prediction parameters.
        
        Returns:
            DataFrame with predictions.
        """
        steps = kwargs.get('steps', 24)
        target_columns = self.metadata.get('target_columns', [])
        
        # Create dummy predictions
        last_date = data.index[-1] if not data.empty else datetime.now()
        if isinstance(last_date, pd.Timestamp):
            last_date = last_date.to_pydatetime()
        
        future_dates = pd.date_range(
            start=last_date + timedelta(hours=1),
            periods=steps,
            freq='H'
        )
        
        result_df = pd.DataFrame(index=future_dates)
        for col in target_columns:
            # Generate some random values as placeholders
            mean = data[col].mean() if col in data.columns else 0
            std = data[col].std() if col in data.columns else 1
            result_df[col] = np.random.normal(mean, std, size=steps)
        
        self.logger.info(f"Generated placeholder predictions for {len(result_df)} time points")
        return result_df
        
    def generate(self, context: Optional[pd.DataFrame] = None, n_samples: int = 1, **kwargs) -> pd.DataFrame:
        """Generate synthetic data using the GAN model.
        
        Placeholder implementation.
        
        Args:
            context: Context data to condition the generation.
            n_samples: Number of samples to generate.
            **kwargs: Additional generation parameters.
        
        Returns:
            DataFrame with generated synthetic data.
        """
        start_date = kwargs.get('start_date', datetime.now())
        target_columns = self.metadata.get('target_columns', [])
        
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        
        # Generate dates
        future_dates = pd.date_range(
            start=start_date,
            periods=n_samples,
            freq='H'
        )
        
        # Generate random data as placeholder
        result_df = pd.DataFrame(index=future_dates)
        for col in target_columns:
            if context is not None and col in context.columns:
                mean = context[col].mean()
                std = context[col].std()
            else:
                mean, std = 0, 1
                
            result_df[col] = np.random.normal(mean, std, size=n_samples)
        
        self.logger.info(f"Generated {len(result_df)} synthetic time points (placeholder)")
        return result_df