import os
import yaml
from typing import Dict, Any, Optional
import logging

class ConfigManager:
    """Configuration manager for the synthetic data generation system.
    
    Handles loading, validating, and providing access to configuration settings.
    Supports dynamic reconfiguration during runtime.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, will look for
                        config.yaml in the current directory or use environment variables.
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.config = {}
        self.load_config()
        
    def load_config(self) -> None:
        """Load configuration from file or environment variables."""
        # Try to load from file first
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                self.logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                self.logger.error(f"Failed to load configuration from {self.config_path}: {e}")
                self.config = {}
        else:
            # Load from environment variables
            self.logger.info("Loading configuration from environment variables")
            self.config = {
                'data_generation': {
                    'seed': int(os.getenv('SYNTH_SEED', '42')),
                    'batch_size': int(os.getenv('SYNTH_BATCH_SIZE', '1000')),
                    'frequency': float(os.getenv('SYNTH_FREQUENCY', '1.0')),  # in seconds
                },
                'models': {
                    'default_model': os.getenv('SYNTH_DEFAULT_MODEL', 'tabular'),
                },
                'drift': {
                    'enabled': os.getenv('SYNTH_DRIFT_ENABLED', 'false').lower() == 'true',
                    'interval': int(os.getenv('SYNTH_DRIFT_INTERVAL', '3600')),  # in seconds
                    'magnitude': float(os.getenv('SYNTH_DRIFT_MAGNITUDE', '0.1')),
                },
                'streaming': {
                    'enabled': os.getenv('SYNTH_STREAMING_ENABLED', 'true').lower() == 'true',
                    'type': os.getenv('SYNTH_STREAMING_TYPE', 'kafka'),
                    'topic': os.getenv('SYNTH_STREAMING_TOPIC', 'synth-data'),
                    'bootstrap_servers': os.getenv('SYNTH_KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
                },
                'api': {
                    'host': os.getenv('SYNTH_API_HOST', '0.0.0.0'),
                    'port': int(os.getenv('SYNTH_API_PORT', '8000')),
                },
                'logging': {
                    'level': os.getenv('SYNTH_LOG_LEVEL', 'INFO'),
                    'format': os.getenv('SYNTH_LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
                },
                'database': {
                    'enabled': os.getenv('SYNTH_DB_ENABLED', 'false').lower() == 'true',
                    'url': os.getenv('SYNTH_DB_URL', 'postgresql://user:password@localhost:5432/synth'),
                },
            }
        
        self.validate_config()
    
    def validate_config(self) -> None:
        """Validate the configuration."""
        # Add validation logic here
        # For now, just check if essential sections exist
        required_sections = ['data_generation', 'models', 'streaming']
        for section in required_sections:
            if section not in self.config:
                self.logger.warning(f"Missing required configuration section: {section}")
                self.config[section] = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.
        
        Args:
            key: Dot-separated path to the configuration value.
            default: Value to return if the key is not found.
            
        Returns:
            The configuration value or the default.
        """
        keys = key.split('.')
        config = self.config
        for k in keys:
            if k not in config:
                return default
            config = config[k]
        return config
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.
        
        Args:
            key: Dot-separated path to the configuration value.
            value: The value to set.
        """
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        self.logger.info(f"Updated configuration: {key} = {value}")
    
    def save(self, path: Optional[str] = None) -> None:
        """Save the configuration to a file.
        
        Args:
            path: Path to save the configuration file. If None, uses the original config path.
        """
        save_path = path or self.config_path
        if not save_path:
            self.logger.warning("No path specified for saving configuration")
            return
        
        try:
            with open(save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            self.logger.info(f"Saved configuration to {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save configuration to {save_path}: {e}")

# Create a singleton instance
config_manager = ConfigManager()