import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
import time
import threading
from queue import Queue

from src.ml.base_model import BaseMLModel
from src.data_sources.base_source import BaseDataSource
from src.config.config_manager import config_manager

class MLDataStreamer:
    """Component that provides continuous data streaming with real and synthetic data.
    
    This class integrates real data sources with ML models to provide a continuous
    stream of data, falling back to synthetic data when real data is unavailable.
    """
    
    def __init__(self, 
                 name: str,
                 data_source: BaseDataSource,
                 ml_model: BaseMLModel,
                 config: Dict[str, Any] = None):
        """Initialize the ML data streamer.
        
        Args:
            name: Name of the streamer.
            data_source: Data source to fetch real data from.
            ml_model: ML model to generate synthetic data.
            config: Configuration parameters for the streamer.
        """
        self.name = name
        self.data_source = data_source
        self.ml_model = ml_model
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Configuration parameters
        self.config.setdefault('streaming_interval_seconds', 60)
        self.config.setdefault('training_interval_minutes', 60)
        self.config.setdefault('batch_size', 100)
        self.config.setdefault('max_buffer_size', 1000)
        self.config.setdefault('fallback_threshold', 0.8)  # % of synthetic data before warning
        
        # Internal state
        self._running = False
        self._stream_thread = None
        self._train_thread = None
        self._buffer = Queue(maxsize=self.config['max_buffer_size'])
        self._real_data_count = 0
        self._synthetic_data_count = 0
        self._last_real_data_time = None
        self._handlers = []
        self._historical_data = pd.DataFrame()
        
    async def initialize(self, 
                    historical_days: int = 30, 
                    params: Dict[str, Any] = None) -> None:
        """Initialize the streamer by training the ML model on historical data.
        
        Args:
            historical_days: Number of days of historical data to fetch.
            params: Parameters for the historical data request.
        """
        self.logger.info(f"Initializing {self.name} with {historical_days} days of historical data")
        
        # Fetch historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=historical_days)
        
        try:
            historical_data = await self.data_source.get_historical_data(
                start_date=start_date,
                end_date=end_date,
                params=params
            )
            
            if historical_data.empty:
                self.logger.warning(f"No historical data found for {self.name}")
                return
                
            self._historical_data = historical_data
            self.logger.info(f"Fetched {len(historical_data)} historical data points")
            
            # Train the ML model on historical data
            target_columns = params.get('target_columns') if params else None
            if not target_columns:
                # Use all numeric columns as targets
                target_columns = [col for col in historical_data.columns 
                                 if pd.api.types.is_numeric_dtype(historical_data[col])]
                
            self.ml_model.train(
                data=historical_data,
                target_columns=target_columns,
                **params if params else {}
            )
            
            self.logger.info(f"Trained ML model on historical data")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize streamer: {e}")
            raise
    
    def add_handler(self, handler: Callable[[pd.DataFrame, Dict[str, Any]], None]) -> None:
        """Add a handler function to process streamed data.
        
        Args:
            handler: Function that takes a DataFrame and metadata dict as input.
        """
        self._handlers.append(handler)
        
    def remove_handler(self, handler: Callable) -> None:
        """Remove a handler function.
        
        Args:
            handler: Handler function to remove.
        """
        if handler in self._handlers:
            self._handlers.remove(handler)
    
    def _call_handlers(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> None:
        """Call all registered handler functions with the provided data.
        
        Args:
            data: DataFrame of data to process.
            metadata: Metadata about the data.
        """
        for handler in self._handlers:
            try:
                handler(data, metadata)
            except Exception as e:
                self.logger.error(f"Error in handler {handler.__name__}: {e}")
    
    async def _fetch_data(self, params: Dict[str, Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fetch data from the data source or generate synthetic data if needed.
        
        Args:
            params: Parameters for the data request.
            
        Returns:
            Tuple of (data DataFrame, metadata dict).
        """
        # Try to fetch real data first
        try:
            data, metadata = await self.data_source.get_data(params)
            
            if not data.empty:
                self._real_data_count += 1
                self._last_real_data_time = datetime.now()
                metadata['data_type'] = 'real'
                return data, metadata
                
        except Exception as e:
            self.logger.warning(f"Failed to fetch real data: {e}")
        
        # If we get here, we need to generate synthetic data
        try:
            # Get context for the synthetic data generation
            context = None
            if not self._historical_data.empty:
                context = self._historical_data.tail(self.config.get('context_window', 100))
                
            # Generate synthetic data
            synthetic_data = self.ml_model.generate(
                context=context,
                n_samples=1,  # Generate one time point
                start_date=datetime.now()
            )
            
            if not synthetic_data.empty:
                self._synthetic_data_count += 1
                
                # Check if we're using too much synthetic data
                total_points = self._real_data_count + self._synthetic_data_count
                if total_points > 10:  # Only check after we have some data points
                    synthetic_ratio = self._synthetic_data_count / total_points
                    if synthetic_ratio > self.config['fallback_threshold']:
                        self.logger.warning(
                            f"Using {synthetic_ratio:.1%} synthetic data, which exceeds the "
                            f"threshold of {self.config['fallback_threshold']:.1%}"
                        )
                
                synthetic_metadata = {
                    'source': f"{self.name}_synthetic",
                    'timestamp': datetime.now().isoformat(),
                    'data_type': 'synthetic',
                    'model_info': self.ml_model.get_info(),
                    'last_real_data_time': (self._last_real_data_time.isoformat() 
                                          if self._last_real_data_time else None)
                }
                
                return synthetic_data, synthetic_metadata
                
        except Exception as e:
            self.logger.error(f"Failed to generate synthetic data: {e}")
            
        # If all else fails, return empty data
        return pd.DataFrame(), {
            'source': self.name,
            'timestamp': datetime.now().isoformat(),
            'data_type': 'empty',
            'error': 'Failed to fetch real or synthetic data'
        }
    
    async def _stream_data_async(self, params: Dict[str, Any] = None) -> None:
        """Stream data continuously in an async loop.
        
        Args:
            params: Parameters for the data requests.
        """
        while self._running:
            try:
                data, metadata = await self._fetch_data(params)
                
                if not data.empty:
                    # Store in historical data for context
                    if not self._historical_data.empty:
                        self._historical_data = pd.concat([self._historical_data, data])
                        # Trim to keep a reasonable size
                        max_history = self.config.get('max_history_size', 10000)
                        if len(self._historical_data) > max_history:
                            self._historical_data = self._historical_data.tail(max_history)
                    else:
                        self._historical_data = data
                        
                    # Add to buffer
                    if not self._buffer.full():
                        self._buffer.put((data, metadata))
                    else:
                        self.logger.warning("Buffer full, dropping oldest data")
                        # Remove oldest item
                        try:
                            self._buffer.get_nowait()
                            self._buffer.put((data, metadata))
                        except:
                            pass
                            
                    # Call handlers directly
                    self._call_handlers(data, metadata)
                    
            except Exception as e:
                self.logger.error(f"Error in streaming loop: {e}")
                
            # Sleep until next interval
            await asyncio.sleep(self.config['streaming_interval_seconds'])
    
    def _stream_data_thread(self, params: Dict[str, Any] = None) -> None:
        """Thread function for streaming data.
        
        Args:
            params: Parameters for the data requests.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._stream_data_async(params))
        finally:
            loop.close()
    
    async def _train_model_async(self, params: Dict[str, Any] = None) -> None:
        """Periodically retrain the ML model on the latest data.
        
        Args:
            params: Parameters for training.
        """
        while self._running:
            try:
                if not self._historical_data.empty:
                    # Train on recent data
                    recent_data = self._historical_data.tail(self.config.get('training_window_size', 1000))
                    
                    target_columns = params.get('target_columns') if params else None
                    if not target_columns:
                        # Use all numeric columns as targets
                        target_columns = [col for col in recent_data.columns 
                                         if pd.api.types.is_numeric_dtype(recent_data[col])]
                    
                    self.ml_model.train(
                        data=recent_data,
                        target_columns=target_columns,
                        **params if params else {}
                    )
                    
                    self.logger.info(f"Retrained ML model on recent data")
                    
            except Exception as e:
                self.logger.error(f"Error in training loop: {e}")
                
            # Sleep until next training interval
            await asyncio.sleep(self.config['training_interval_minutes'] * 60)
    
    def _train_model_thread(self, params: Dict[str, Any] = None) -> None:
        """Thread function for model training.
        
        Args:
            params: Parameters for training.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._train_model_async(params))
        finally:
            loop.close()
    
    def start(self, params: Dict[str, Any] = None) -> None:
        """Start streaming data.
        
        Args:
            params: Parameters for the data requests and training.
        """
        if self._running:
            self.logger.warning(f"{self.name} is already running")
            return
            
        self._running = True
        
        # Start streaming thread
        self._stream_thread = threading.Thread(
            target=self._stream_data_thread,
            args=(params,),
            daemon=True
        )
        self._stream_thread.start()
        
        # Start training thread
        self._train_thread = threading.Thread(
            target=self._train_model_thread,
            args=(params,),
            daemon=True
        )
        self._train_thread.start()
        
        self.logger.info(f"Started {self.name} streaming")
    
    def stop(self) -> None:
        """Stop streaming data."""
        if not self._running:
            self.logger.warning(f"{self.name} is not running")
            return
            
        self._running = False
        
        # Threads will terminate on their own since they check self._running
        self.logger.info(f"Stopped {self.name} streaming")
    
    def get_status(self) -> Dict[str, Any]:
        """Get the status of the streamer.
        
        Returns:
            Dictionary with status information.
        """
        synthetic_ratio = 0
        total_points = self._real_data_count + self._synthetic_data_count
        if total_points > 0:
            synthetic_ratio = self._synthetic_data_count / total_points
            
        return {
            'name': self.name,
            'running': self._running,
            'real_data_count': self._real_data_count,
            'synthetic_data_count': self._synthetic_data_count,
            'synthetic_ratio': synthetic_ratio,
            'last_real_data_time': (self._last_real_data_time.isoformat() 
                                  if self._last_real_data_time else None),
            'buffer_size': self._buffer.qsize(),
            'buffer_capacity': self._buffer.maxsize,
            'model_info': self.ml_model.get_info(),
            'source_info': self.data_source.get_source_info()
        }
    
    async def get_data_snapshot(self, n_points: int = 100) -> pd.DataFrame:
        """Get a snapshot of the most recent data.
        
        Args:
            n_points: Number of data points to return.
            
        Returns:
            DataFrame with the most recent data.
        """
        if self._historical_data.empty:
            return pd.DataFrame()
            
        return self._historical_data.tail(n_points)