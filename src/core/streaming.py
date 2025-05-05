import json
import logging
import time
import threading
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import pandas as pd

from src.config.config_manager import config_manager

class BaseStreamer(ABC):
    """Base class for all data streamers.
    
    Provides common functionality and interface for streaming synthetic data.
    """
    
    def __init__(self, name: str):
        """Initialize the streamer.
        
        Args:
            name: Name of the streamer.
        """
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.running = False
        self.thread = None
    
    @abstractmethod
    def send(self, data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> bool:
        """Send data to the stream.
        
        Args:
            data: The data to send.
            
        Returns:
            True if successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the streamer and release resources."""
        pass
    
    def metadata(self) -> Dict[str, Any]:
        """Get metadata about the streamer.
        
        Returns:
            Dictionary with metadata.
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "running": self.running,
        }


class KafkaStreamer(BaseStreamer):
    """Streamer for Apache Kafka.
    
    Streams data to a Kafka topic.
    """
    
    def __init__(self, name: str, topic: str, bootstrap_servers: str):
        """Initialize the Kafka streamer.
        
        Args:
            name: Name of the streamer.
            topic: Kafka topic to stream to.
            bootstrap_servers: Kafka bootstrap servers.
        """
        super().__init__(name)
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers
        self.producer = None
        self.logger.info(f"Initialized Kafka streamer to topic {topic}")
        
        self._initialize_producer()
    
    def _initialize_producer(self) -> None:
        """Initialize the Kafka producer."""
        try:
            from confluent_kafka import Producer
            
            config = {
                'bootstrap.servers': self.bootstrap_servers,
                'client.id': f'synth-producer-{self.name}'
            }
            
            self.producer = Producer(config)
            self.logger.info("Kafka producer initialized")
        except ImportError:
            self.logger.error("confluent-kafka package is required for KafkaStreamer")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize Kafka producer: {e}")
            raise
    
    def send(self, data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> bool:
        """Send data to the Kafka topic.
        
        Args:
            data: The data to send. If DataFrame, each row will be sent as a separate message.
            
        Returns:
            True if successful, False otherwise.
        """
        if self.producer is None:
            self.logger.error("Kafka producer not initialized")
            return False
        
        try:
            # Convert DataFrame to list of records
            if isinstance(data, pd.DataFrame):
                records = data.to_dict(orient='records')
            elif isinstance(data, dict):
                records = [data]
            else:
                records = data
            
            for record in records:
                # Convert to JSON string
                value = json.dumps(record, default=str).encode('utf-8')
                
                # Send to Kafka
                self.producer.produce(self.topic, value=value)
            
            # Flush to ensure delivery
            self.producer.flush()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send data to Kafka: {e}")
            return False
    
    def close(self) -> None:
        """Close the Kafka producer."""
        if self.producer is not None:
            self.producer.flush()
            self.logger.info("Closed Kafka producer")
            self.producer = None


class ConsoleStreamer(BaseStreamer):
    """Simple streamer that outputs to console.
    
    Useful for testing and debugging.
    """
    
    def __init__(self, name: str, pretty_print: bool = True):
        """Initialize the console streamer.
        
        Args:
            name: Name of the streamer.
            pretty_print: Whether to pretty-print the data.
        """
        super().__init__(name)
        self.pretty_print = pretty_print
    
    def send(self, data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> bool:
        """Print data to the console.
        
        Args:
            data: The data to print.
            
        Returns:
            True (always successful).
        """
        try:
            if isinstance(data, pd.DataFrame):
                if self.pretty_print:
                    print(f"\n===== {self.name} Output =====")
                    print(data.to_string())
                    print("==========================\n")
                else:
                    print(data.to_dict(orient='records'))
            else:
                if self.pretty_print:
                    print(f"\n===== {self.name} Output =====")
                    print(json.dumps(data, indent=2, default=str))
                    print("==========================\n")
                else:
                    print(json.dumps(data, default=str))
            return True
        except Exception as e:
            self.logger.error(f"Failed to print data: {e}")
            return False
    
    def close(self) -> None:
        """Close the console streamer (no-op)."""
        pass


class FileStreamer(BaseStreamer):
    """Streamer that writes data to a file.
    
    Supports various file formats including CSV, JSON, and Parquet.
    """
    
    def __init__(self, name: str, file_path: str, format: str = 'csv', mode: str = 'a'):
        """Initialize the file streamer.
        
        Args:
            name: Name of the streamer.
            file_path: Path to the output file.
            format: File format ('csv', 'json', or 'parquet').
            mode: File mode ('w' for write, 'a' for append).
        """
        super().__init__(name)
        self.file_path = file_path
        self.format = format.lower()
        self.mode = mode
        
        if self.format not in ['csv', 'json', 'parquet']:
            raise ValueError(f"Unsupported file format: {format}")
        
        self.logger.info(f"Initialized file streamer to {file_path} ({format})")
    
    def send(self, data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> bool:
        """Write data to the file.
        
        Args:
            data: The data to write.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Convert to DataFrame if needed
            if not isinstance(data, pd.DataFrame):
                if isinstance(data, dict):
                    data = pd.DataFrame([data])
                else:
                    data = pd.DataFrame(data)
            
            # Write to file based on format
            if self.format == 'csv':
                # Check if file exists for header
                write_header = not (self.mode == 'a' and os.path.exists(self.file_path))
                data.to_csv(self.file_path, mode=self.mode, header=write_header, index=False)
            elif self.format == 'json':
                # For JSON in append mode, we need to handle things differently
                if self.mode == 'a' and os.path.exists(self.file_path):
                    with open(self.file_path, 'a') as f:
                        for _, row in data.iterrows():
                            f.write(json.dumps(row.to_dict(), default=str) + '\n')
                else:
                    # Write as newline-delimited JSON
                    with open(self.file_path, 'w') as f:
                        for _, row in data.iterrows():
                            f.write(json.dumps(row.to_dict(), default=str) + '\n')
            elif self.format == 'parquet':
                # For parquet, append mode requires special handling
                try:
                    import pyarrow as pa
                    import pyarrow.parquet as pq
                    
                    if self.mode == 'a' and os.path.exists(self.file_path):
                        # Read existing and append
                        existing_data = pq.read_table(self.file_path).to_pandas()
                        combined = pd.concat([existing_data, data], ignore_index=True)
                        pq.write_table(pa.Table.from_pandas(combined), self.file_path)
                    else:
                        # Write new file
                        pq.write_table(pa.Table.from_pandas(data), self.file_path)
                except ImportError:
                    self.logger.error("pyarrow package is required for Parquet support")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to write data to file: {e}")
            return False
    
    def close(self) -> None:
        """Close the file streamer (no-op, files are closed after each write)."""
        pass


class StreamingManager:
    """Manager for data streaming.
    
    Coordinates the streaming of data to multiple destinations.
    """
    
    def __init__(self):
        """Initialize the streaming manager."""
        self.logger = logging.getLogger(__name__)
        self.streamers = {}
    
    def add_streamer(self, streamer: BaseStreamer) -> None:
        """Add a streamer to the manager.
        
        Args:
            streamer: The streamer to add.
        """
        self.streamers[streamer.name] = streamer
        self.logger.info(f"Added streamer: {streamer.name}")
    
    def remove_streamer(self, name: str) -> None:
        """Remove a streamer from the manager.
        
        Args:
            name: Name of the streamer to remove.
        """
        if name in self.streamers:
            self.streamers[name].close()
            del self.streamers[name]
            self.logger.info(f"Removed streamer: {name}")
    
    def stream(self, data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, bool]:
        """Stream data to all registered streamers.
        
        Args:
            data: The data to stream.
            
        Returns:
            Dictionary mapping streamer names to success status.
        """
        results = {}
        
        for name, streamer in self.streamers.items():
            results[name] = streamer.send(data)
            
        return results
    
    def close_all(self) -> None:
        """Close all streamers."""
        for name, streamer in self.streamers.items():
            streamer.close()
            self.logger.info(f"Closed streamer: {name}")
        self.streamers = {}

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
import time
import threading
import json
import os
from collections import defaultdict

from src.ml.ml_streamer import MLDataStreamer
from src.data_sources.base_source import BaseDataSource
from src.ml.base_model import BaseMLModel
from src.config.config_manager import config_manager

class StreamingEngine:
    """Main engine for streaming data collection and distribution.
    
    This class manages multiple data streamers and provides a unified interface
    for consuming streaming data from various sources.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the streaming engine.
        
        Args:
            config_path: Path to configuration file or directory.
        """
        self.logger = logging.getLogger(__name__)
        self.streamers = {}
        self.listeners = {}
        self._initialized = False
        self._running = False
        self._monitors = {}
        
        # Load configuration if provided
        if config_path:
            if os.path.isfile(config_path):
                self._load_config_file(config_path)
            elif os.path.isdir(config_path):
                for filename in os.listdir(config_path):
                    if filename.endswith('.json'):
                        file_path = os.path.join(config_path, filename)
                        self._load_config_file(file_path)
    
    def _load_config_file(self, config_path: str):
        """Load configuration from a file.
        
        Args:
            config_path: Path to the configuration file.
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Process configuration
            if 'streamers' in config:
                for streamer_config in config['streamers']:
                    try:
                        name = streamer_config.get('name')
                        if not name:
                            self.logger.warning(f"Skipping streamer config without name: {streamer_config}")
                            continue
                            
                        self.logger.info(f"Found streamer configuration for {name}")
                        
                        # Store configuration for later initialization
                        self.streamers[name] = {
                            'config': streamer_config,
                            'streamer': None
                        }
                        
                    except Exception as e:
                        self.logger.error(f"Error processing streamer config {streamer_config}: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error loading config from {config_path}: {e}")
    
    async def initialize(self):
        """Initialize all streamers based on configuration.
        
        This initializes data sources, ML models, and MLDataStreamers for all 
        configured data streams.
        """
        if self._initialized:
            self.logger.warning("StreamingEngine already initialized")
            return
            
        self.logger.info("Initializing StreamingEngine")
        
        # Create data sources, models, and streamers
        for name, streamer_info in list(self.streamers.items()):
            try:
                config = streamer_info['config']
                
                # Create data source
                source_config = config.get('data_source', {})
                source_type = source_config.get('type')
                source_name = source_config.get('name', f"{name}_source")
                
                data_source = self._create_data_source(source_type, source_name, source_config)
                if not data_source:
                    self.logger.error(f"Failed to create data source for {name}")
                    continue
                
                # Create ML model
                model_config = config.get('ml_model', {})
                model_type = model_config.get('type')
                model_name = model_config.get('name', f"{name}_model")
                
                ml_model = self._create_ml_model(model_type, model_name, model_config)
                if not ml_model:
                    self.logger.error(f"Failed to create ML model for {name}")
                    continue
                
                # Create streamer
                streamer_config = config.get('streamer_config', {})
                
                streamer = MLDataStreamer(
                    name=name,
                    data_source=data_source,
                    ml_model=ml_model,
                    config=streamer_config
                )
                
                # Initialize with historical data
                historical_days = config.get('historical_days', 30)
                
                try:
                    await streamer.initialize(
                        historical_days=historical_days,
                        params=config.get('initialization_params', {})
                    )
                except Exception as e:
                    self.logger.error(f"Error initializing streamer {name}: {e}")
                    # Continue with the streamer even if initialization fails
                
                # Store the streamer
                streamer_info['streamer'] = streamer
                
                # Set up monitoring if configured
                if config.get('monitor', False):
                    self._setup_monitoring(name, streamer)
                
                self.logger.info(f"Successfully created streamer {name}")
                
            except Exception as e:
                self.logger.error(f"Error setting up streamer {name}: {e}")
        
        self._initialized = True
        self.logger.info("StreamingEngine initialization complete")
    
    def _create_data_source(self, source_type: str, name: str, config: Dict[str, Any]) -> Optional[BaseDataSource]:
        """Create a data source based on configuration.
        
        Args:
            source_type: Type of data source to create.
            name: Name for the data source.
            config: Configuration for the data source.
            
        Returns:
            Instantiated data source or None if creation failed.
        """
        try:
            # Import the appropriate class based on the source type
            if source_type == 'alpha_vantage':
                from src.data_sources.market_sources import AlphaVantageSource
                return AlphaVantageSource(name, config)
                
            elif source_type == 'yahoo_finance':
                from src.data_sources.market_sources import YahooFinanceSource
                return YahooFinanceSource(name, config)
                
            elif source_type == 'cryptocompare':
                from src.data_sources.market_sources import CryptoCompareSource
                return CryptoCompareSource(name, config)
                
            else:
                self.logger.error(f"Unknown data source type: {source_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating data source {name} of type {source_type}: {e}")
            return None
    
    def _create_ml_model(self, model_type: str, name: str, config: Dict[str, Any]) -> Optional[BaseMLModel]:
        """Create an ML model based on configuration.
        
        Args:
            model_type: Type of ML model to create.
            name: Name for the ML model.
            config: Configuration for the ML model.
            
        Returns:
            Instantiated ML model or None if creation failed.
        """
        try:
            # Import the appropriate class based on the model type
            if model_type == 'time_series':
                from src.ml.timeseries_models import TimeSeriesModel
                return TimeSeriesModel(name, config)
                
            # Add other model types as needed
                
            else:
                self.logger.error(f"Unknown ML model type: {model_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating ML model {name} of type {model_type}: {e}")
            return None
    
    def _setup_monitoring(self, name: str, streamer: MLDataStreamer):
        """Set up monitoring for a streamer.
        
        Args:
            name: Name of the streamer.
            streamer: MLDataStreamer instance.
        """
        def monitor_handler(data: pd.DataFrame, metadata: Dict[str, Any]):
            # Log basic information about the data
            data_type = metadata.get('data_type', 'unknown')
            
            if data_type == 'real':
                self.logger.info(f"[{name}] Received real data with {len(data)} rows")
            elif data_type == 'synthetic':
                self.logger.info(f"[{name}] Generated synthetic data with {len(data)} rows")
            
            # Store monitoring data
            if name not in self._monitors:
                self._monitors[name] = {
                    'data_points': 0,
                    'real_data': 0,
                    'synthetic_data': 0,
                    'errors': 0,
                    'last_data_time': None,
                    'data_types': defaultdict(int)
                }
                
            monitor = self._monitors[name]
            monitor['data_points'] += len(data)
            monitor['last_data_time'] = metadata.get('timestamp')
            monitor['data_types'][data_type] += 1
            
            if data_type == 'real':
                monitor['real_data'] += len(data)
            elif data_type == 'synthetic':
                monitor['synthetic_data'] += len(data)
                
            if 'error' in metadata and metadata['error']:
                monitor['errors'] += 1
        
        # Add the monitoring handler
        streamer.add_handler(monitor_handler)
        self.logger.info(f"Set up monitoring for streamer {name}")
        
    def register_listener(self, name: str, listener: Callable[[pd.DataFrame, Dict[str, Any]], None]) -> bool:
        """Register a listener for streamed data.
        
        Args:
            name: Name of the streamer to listen to.
            listener: Callable that receives a DataFrame and metadata dictionary.
            
        Returns:
            True if registration was successful, False otherwise.
        """
        if name not in self.streamers or not self.streamers[name].get('streamer'):
            self.logger.error(f"Cannot register listener for unknown streamer: {name}")
            return False
            
        streamer = self.streamers[name]['streamer']
        streamer.add_handler(listener)
        
        # Store the listener for reference
        if name not in self.listeners:
            self.listeners[name] = []
            
        self.listeners[name].append(listener)
        return True
    
    def unregister_listener(self, name: str, listener: Callable) -> bool:
        """Unregister a listener from streamed data.
        
        Args:
            name: Name of the streamer.
            listener: Listener to unregister.
            
        Returns:
            True if unregistration was successful, False otherwise.
        """
        if name not in self.streamers or not self.streamers[name].get('streamer'):
            self.logger.error(f"Cannot unregister listener for unknown streamer: {name}")
            return False
            
        streamer = self.streamers[name]['streamer']
        streamer.remove_handler(listener)
        
        # Remove from stored listeners
        if name in self.listeners and listener in self.listeners[name]:
            self.listeners[name].remove(listener)
            
        return True
    
    def start(self):
        """Start all streamers."""
        if not self._initialized:
            self.logger.error("Cannot start StreamingEngine before initialization")
            return
            
        if self._running:
            self.logger.warning("StreamingEngine is already running")
            return
            
        self.logger.info("Starting StreamingEngine")
        
        for name, streamer_info in self.streamers.items():
            streamer = streamer_info.get('streamer')
            if streamer:
                try:
                    streamer.start(params=streamer_info['config'].get('stream_params', {}))
                    self.logger.info(f"Started streamer {name}")
                except Exception as e:
                    self.logger.error(f"Error starting streamer {name}: {e}")
        
        self._running = True
    
    def stop(self):
        """Stop all streamers."""
        if not self._running:
            self.logger.warning("StreamingEngine is not running")
            return
            
        self.logger.info("Stopping StreamingEngine")
        
        for name, streamer_info in self.streamers.items():
            streamer = streamer_info.get('streamer')
            if streamer:
                try:
                    streamer.stop()
                    self.logger.info(f"Stopped streamer {name}")
                except Exception as e:
                    self.logger.error(f"Error stopping streamer {name}: {e}")
        
        self._running = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get status information for all streamers.
        
        Returns:
            Dictionary with status information.
        """
        status = {
            'initialized': self._initialized,
            'running': self._running,
            'streamers': {},
            'monitors': {},
        }
        
        for name, streamer_info in self.streamers.items():
            streamer = streamer_info.get('streamer')
            if streamer:
                try:
                    status['streamers'][name] = streamer.get_status()
                except Exception as e:
                    status['streamers'][name] = {'error': str(e)}
        
        # Include monitoring data if available
        status['monitors'] = self._monitors
        
        return status
    
    async def get_data_snapshot(self, name: str, n_points: int = 100) -> Optional[pd.DataFrame]:
        """Get a snapshot of recent data from a streamer.
        
        Args:
            name: Name of the streamer.
            n_points: Number of data points to return.
            
        Returns:
            DataFrame with recent data or None if the streamer doesn't exist.
        """
        if name not in self.streamers or not self.streamers[name].get('streamer'):
            self.logger.error(f"Cannot get data from unknown streamer: {name}")
            return None
            
        streamer = self.streamers[name]['streamer']
        return await streamer.get_data_snapshot(n_points)
    
    def add_streamer(self, name: str, config: Dict[str, Any]) -> bool:
        """Add a new streamer configuration.
        
        Args:
            name: Name for the new streamer.
            config: Configuration for the streamer.
            
        Returns:
            True if the streamer was added, False otherwise.
        """
        if name in self.streamers:
            self.logger.warning(f"Streamer {name} already exists")
            return False
            
        self.streamers[name] = {
            'config': config,
            'streamer': None
        }
        
        # Initialize the streamer if the engine is already initialized
        if self._initialized:
            asyncio.create_task(self._initialize_streamer(name))
            
        return True
    
    async def _initialize_streamer(self, name: str) -> bool:
        """Initialize a single streamer.
        
        Args:
            name: Name of the streamer to initialize.
            
        Returns:
            True if initialization was successful, False otherwise.
        """
        try:
            streamer_info = self.streamers.get(name)
            if not streamer_info:
                self.logger.error(f"Cannot initialize unknown streamer: {name}")
                return False
                
            config = streamer_info['config']
            
            # Create data source
            source_config = config.get('data_source', {})
            source_type = source_config.get('type')
            source_name = source_config.get('name', f"{name}_source")
            
            data_source = self._create_data_source(source_type, source_name, source_config)
            if not data_source:
                self.logger.error(f"Failed to create data source for {name}")
                return False
            
            # Create ML model
            model_config = config.get('ml_model', {})
            model_type = model_config.get('type')
            model_name = model_config.get('name', f"{name}_model")
            
            ml_model = self._create_ml_model(model_type, model_name, model_config)
            if not ml_model:
                self.logger.error(f"Failed to create ML model for {name}")
                return False
            
            # Create streamer
            streamer_config = config.get('streamer_config', {})
            
            streamer = MLDataStreamer(
                name=name,
                data_source=data_source,
                ml_model=ml_model,
                config=streamer_config
            )
            
            # Initialize with historical data
            historical_days = config.get('historical_days', 30)
            
            try:
                await streamer.initialize(
                    historical_days=historical_days,
                    params=config.get('initialization_params', {})
                )
            except Exception as e:
                self.logger.error(f"Error initializing streamer {name}: {e}")
                # Continue with the streamer even if initialization fails
            
            # Store the streamer
            streamer_info['streamer'] = streamer
            
            # Set up monitoring if configured
            if config.get('monitor', False):
                self._setup_monitoring(name, streamer)
            
            # Start the streamer if the engine is running
            if self._running:
                streamer.start(params=config.get('stream_params', {}))
            
            self.logger.info(f"Successfully initialized streamer {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing streamer {name}: {e}")
            return False
    
    def remove_streamer(self, name: str) -> bool:
        """Remove a streamer.
        
        Args:
            name: Name of the streamer to remove.
            
        Returns:
            True if the streamer was removed, False otherwise.
        """
        if name not in self.streamers:
            self.logger.warning(f"Cannot remove unknown streamer: {name}")
            return False
            
        # Stop the streamer if it's running
        streamer = self.streamers[name].get('streamer')
        if streamer:
            try:
                streamer.stop()
            except Exception as e:
                self.logger.error(f"Error stopping streamer {name}: {e}")
        
        # Remove all listeners
        if name in self.listeners:
            del self.listeners[name]
            
        # Remove from monitors
        if name in self._monitors:
            del self._monitors[name]
            
        # Remove the streamer
        del self.streamers[name]
        
        self.logger.info(f"Removed streamer {name}")
        return True