# Synth: Production-Ready Synthetic Data Generation System

Synth is a powerful, flexible synthetic data generation system designed for producing realistic data at scale. The system can generate and continuously stream complex, realistic data—including controlled drift events—into downstream applications and data pipelines.

## Features

- **Flexible Data Generation**: Generate tabular or time series data with customizable schemas
- **Realistic Data**: Create data that mimics real-world patterns and distributions
- **Controlled Drift Simulation**: Introduce and control data drift to test ML model resilience
- **Multiple Output Formats**: Output to console, files (CSV, JSON, Parquet), or Kafka
- **Continuous Streaming**: Generate data in continuous mode with configurable frequency
- **REST API**: Control the system through a RESTful API
- **Command-line Interface**: Easy-to-use CLI for quick operations
- **Observability**: Monitor performance and operational metrics
- **Dynamic Reconfiguration**: Update system parameters without restarting

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/synth.git
cd synth

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

### Generate Example Schema

```bash
# Create example tabular data schema
python main.py schema create-example tabular --output tabular_schema.json

# Create example time series schema
python main.py schema create-example timeseries --output timeseries_schema.json
```

### Generate Data

```bash
# Generate tabular data to console
python main.py run --schema tabular_schema.json --type tabular

# Generate time series data to a CSV file
python main.py run --schema timeseries_schema.json --type timeseries --output data.csv

# Generate continuous data with drift
python main.py run --schema tabular_schema.json --type tabular --continuous --drift --drift-interval 60 --drift-magnitude 0.2
```

### Run API Server

```bash
# Start the API server
python main.py api
```

The API server will start on http://localhost:8000 by default.

## Configuration

You can configure the system through environment variables, command-line arguments, or the configuration API.

### Environment Variables

- `SYNTH_SEED`: Random seed for reproducibility (default: 42)
- `SYNTH_BATCH_SIZE`: Number of records in each batch (default: 1000)
- `SYNTH_FREQUENCY`: Generation frequency in seconds (default: 1.0)
- `SYNTH_DRIFT_ENABLED`: Enable drift simulation (default: false)
- `SYNTH_DRIFT_INTERVAL`: Interval between drift events in seconds (default: 3600)
- `SYNTH_DRIFT_MAGNITUDE`: Magnitude of drift (default: 0.1)
- `SYNTH_API_HOST`: Host for the API server (default: 0.0.0.0)
- `SYNTH_API_PORT`: Port for the API server (default: 8000)
- `SYNTH_LOG_LEVEL`: Logging level (default: INFO)

### Configuration File

You can also use a YAML configuration file:

```yaml
data_generation:
  seed: 42
  batch_size: 1000
  frequency: 1.0
models:
  default_model: tabular
drift:
  enabled: false
  interval: 3600
  magnitude: 0.1
streaming:
  enabled: true
  type: console
api:
  host: 0.0.0.0
  port: 8000
logging:
  level: INFO
```

## Data Schema Definition

### Tabular Data Schema

```json
{
  "id": {
    "type": "int",
    "min": 1000,
    "max": 9999
  },
  "name": {
    "type": "name"
  },
  "email": {
    "type": "email"
  },
  "age": {
    "type": "int",
    "min": 18,
    "max": 90
  },
  "income": {
    "type": "float",
    "mean": 50000,
    "std": 20000,
    "min": 0
  },
  "created_at": {
    "type": "datetime",
    "start_date": "2020-01-01",
    "end_date": "2023-12-31"
  },
  "status": {
    "type": "category",
    "categories": ["active", "inactive", "pending"],
    "probabilities": [0.7, 0.2, 0.1]
  }
}
```

### Time Series Schema

```json
{
  "time_field": "timestamp",
  "time_interval_seconds": 300,
  "value": {
    "type": "float",
    "mean": 100,
    "std": 15,
    "min": 0
  },
  "count": {
    "type": "int",
    "min": 0,
    "max": 1000
  },
  "category": {
    "type": "category",
    "categories": ["A", "B", "C", "D"],
    "probabilities": [0.4, 0.3, 0.2, 0.1]
  },
  "trends": {
    "value": {
      "type": "linear",
      "strength": 0.1
    }
  },
  "seasonality": {
    "value": {
      "period": 24,
      "amplitude": 0.2
    }
  }
}
```

## REST API

The system provides a RESTful API for managing data generation.

### Endpoints

- `GET /status`: Get system status
- `POST /generators`: Add a data generator
- `GET /generators`: List all generators
- `DELETE /generators/{name}`: Remove a generator
- `POST /streamers`: Add a data streamer
- `GET /streamers`: List all streamers
- `DELETE /streamers/{name}`: Remove a streamer
- `POST /generate`: Generate a single batch of data
- `POST /start`: Start continuous data generation
- `POST /stop`: Stop continuous data generation
- `POST /config`: Update system configuration

## Architecture

The system is designed with a modular architecture:

- **Core Engine**: Central component that orchestrates the system
- **Data Generators**: Components that generate different types of synthetic data
- **Streamers**: Components that output data to different destinations
- **Configuration Manager**: Handles system configuration
- **API Server**: Provides RESTful access to the system
- **CLI**: Command-line interface for the system

## Extending the System

### Adding New Data Generators

You can create new data generators by extending the `BaseDataGenerator` class:

```python
from src.core.data_generator import BaseDataGenerator

class MyCustomGenerator(BaseDataGenerator):
    def __init__(self, name, schema):
        super().__init__(name, schema)
        # Custom initialization
        
    def generate_batch(self, size):
        # Custom data generation logic
        return data
        
    def apply_drift(self, data, magnitude):
        # Custom drift logic
        return drifted_data
```

### Adding New Streamers

You can create new streamers by extending the `BaseStreamer` class:

```python
from src.core.streaming import BaseStreamer

class MyCustomStreamer(BaseStreamer):
    def __init__(self, name, **kwargs):
        super().__init__(name)
        # Custom initialization
        
    def send(self, data):
        # Custom data sending logic
        return success
        
    def close(self):
        # Custom cleanup logic
        pass
```

## Future Enhancements

- Support for more complex data types (e.g., images, text)
- Integration with additional streaming platforms
- More advanced drift patterns
- Web-based UI for configuration and monitoring
- Distributed operation for higher throughput

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
