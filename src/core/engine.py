import logging
import time
import threading
from typing import Dict, Any, Optional, List, Union, Callable
import pandas as pd
from datetime import datetime, timedelta

from src.config.config_manager import config_manager
from src.core.streaming import StreamingManager, BaseStreamer
from src.core.data_generator import BaseDataGenerator

class SynthEngine:
    """Core engine for the synthetic data generation system.
    
    Orchestrates data generation, drift simulation, and streaming.
    """
    
    def __init__(self):
        """Initialize the synthetic data engine."""
        self.logger = logging.getLogger(__name__)
        self.generators = {}
        self.streaming_manager = StreamingManager()
        self.running = False
        self.thread = None
        self.stop_event = threading.Event()
        
        # Drift configuration
        self.drift_enabled = config_manager.get('drift.enabled', False)
        self.drift_interval = config_manager.get('drift.interval', 3600)  # seconds
        self.drift_magnitude = config_manager.get('drift.magnitude', 0.1)
        
        # Generation configuration
        self.batch_size = config_manager.get('data_generation.batch_size', 1000)
        self.generation_frequency = config_manager.get('data_generation.frequency', 1.0)  # seconds
        
        # Monitoring
        self.stats = {
            'batches_generated': 0,
            'records_generated': 0,
            'drift_events': 0,
            'start_time': None,
            'last_drift_time': None,
        }
        
        self.logger.info("Initialized SynthEngine")
    
    def add_generator(self, generator: BaseDataGenerator) -> None:
        """Add a data generator to the engine.
        
        Args:
            generator: The generator to add.
        """
        self.generators[generator.name] = generator
        self.logger.info(f"Added generator: {generator.name}")
    
    def remove_generator(self, name: str) -> None:
        """Remove a generator from the engine.
        
        Args:
            name: Name of the generator to remove.
        """
        if name in self.generators:
            del self.generators[name]
            self.logger.info(f"Removed generator: {name}")
    
    def add_streamer(self, streamer: BaseStreamer) -> None:
        """Add a streamer to the engine.
        
        Args:
            streamer: The streamer to add.
        """
        self.streaming_manager.add_streamer(streamer)
    
    def remove_streamer(self, name: str) -> None:
        """Remove a streamer from the engine.
        
        Args:
            name: Name of the streamer to remove.
        """
        self.streaming_manager.remove_streamer(name)
    
    def generate_batch(self, generator_name: Optional[str] = None, apply_drift: bool = False) -> Dict[str, pd.DataFrame]:
        """Generate a batch of data from a specific generator or all generators.
        
        Args:
            generator_name: Name of the generator to use. If None, use all generators.
            apply_drift: Whether to apply drift to the generated data.
            
        Returns:
            Dictionary mapping generator names to generated data.
        """
        result = {}
        
        generators_to_use = {}
        if generator_name is not None:
            if generator_name in self.generators:
                generators_to_use = {generator_name: self.generators[generator_name]}
            else:
                self.logger.error(f"Generator not found: {generator_name}")
                return {}
        else:
            generators_to_use = self.generators
        
        for name, generator in generators_to_use.items():
            try:
                data = generator.generate_batch(self.batch_size)
                
                if apply_drift:
                    data = generator.apply_drift(data, self.drift_magnitude)
                    self.logger.info(f"Applied drift with magnitude {self.drift_magnitude} to {name}")
                
                result[name] = data
                
                self.stats['batches_generated'] += 1
                self.stats['records_generated'] += len(data)
                
                if apply_drift:
                    self.stats['drift_events'] += 1
                    self.stats['last_drift_time'] = datetime.now()
                    
            except Exception as e:
                self.logger.error(f"Error generating batch from {name}: {e}")
        
        return result
    
    def stream_batch(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, bool]]:
        """Stream a batch of data from multiple generators.
        
        Args:
            data: Dictionary mapping generator names to generated data.
            
        Returns:
            Dictionary mapping generator names to streaming results.
        """
        results = {}
        
        for generator_name, df in data.items():
            # Add generator name as metadata
            df['generator'] = generator_name
            df['timestamp'] = datetime.now().isoformat()
            
            # Stream the data
            stream_result = self.streaming_manager.stream(df)
            results[generator_name] = stream_result
            
            self.logger.debug(f"Streamed batch from {generator_name} ({len(df)} records)")
        
        return results
    
    def run_once(self, apply_drift: bool = False) -> Dict[str, Dict[str, bool]]:
        """Generate and stream one batch of data.
        
        Args:
            apply_drift: Whether to apply drift to the generated data.
            
        Returns:
            The results of streaming the generated data.
        """
        data = self.generate_batch(apply_drift=apply_drift)
        if not data:
            return {}
        
        return self.stream_batch(data)
    
    def run_continuous(self) -> None:
        """Run the engine continuously in a separate thread."""
        if self.running:
            self.logger.warning("Engine is already running")
            return
        
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._continuous_run)
        self.thread.daemon = True
        self.thread.start()
        self.running = True
        self.stats['start_time'] = datetime.now()
        self.logger.info("Started continuous data generation")
    
    def _continuous_run(self) -> None:
        """Internal method for continuous data generation."""
        last_drift_time = time.time()
        
        while not self.stop_event.is_set():
            try:
                current_time = time.time()
                
                # Check if we should apply drift
                apply_drift = False
                if self.drift_enabled and (current_time - last_drift_time) >= self.drift_interval:
                    apply_drift = True
                    last_drift_time = current_time
                
                # Generate and stream a batch
                self.run_once(apply_drift=apply_drift)
                
                # Sleep until next batch
                time.sleep(self.generation_frequency)
                
            except Exception as e:
                self.logger.error(f"Error in continuous run: {e}")
                time.sleep(1)  # Prevent tight error loop
    
    def stop(self) -> None:
        """Stop the continuous data generation."""
        if not self.running:
            self.logger.warning("Engine is not running")
            return
        
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5.0)
        self.running = False
        self.logger.info("Stopped continuous data generation")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the engine.
        
        Returns:
            Dictionary with statistics.
        """
        stats = self.stats.copy()
        
        # Calculate runtime if started
        if stats['start_time']:
            runtime = datetime.now() - stats['start_time']
            stats['runtime'] = str(runtime)
            
            # Calculate records per second
            if runtime.total_seconds() > 0:
                stats['records_per_second'] = stats['records_generated'] / runtime.total_seconds()
        
        return stats
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Update the engine configuration.
        
        Args:
            config: Dictionary with configuration settings.
        """
        if 'batch_size' in config:
            self.batch_size = config['batch_size']
            self.logger.info(f"Updated batch_size to {self.batch_size}")
        
        if 'generation_frequency' in config:
            self.generation_frequency = config['generation_frequency']
            self.logger.info(f"Updated generation_frequency to {self.generation_frequency}")
        
        if 'drift_enabled' in config:
            self.drift_enabled = config['drift_enabled']
            self.logger.info(f"Updated drift_enabled to {self.drift_enabled}")
            
        if 'drift_interval' in config:
            self.drift_interval = config['drift_interval']
            self.logger.info(f"Updated drift_interval to {self.drift_interval}")
            
        if 'drift_magnitude' in config:
            self.drift_magnitude = config['drift_magnitude']
            self.logger.info(f"Updated drift_magnitude to {self.drift_magnitude}")
    
    def close(self) -> None:
        """Close the engine and release resources."""
        if self.running:
            self.stop()
        
        self.streaming_manager.close_all()
        self.logger.info("Engine resources released")