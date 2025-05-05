#!/usr/bin/env python
import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, Any, Optional

from src.config.config_manager import config_manager
from src.core.engine import SynthEngine
from src.models.data_generators import TabularDataGenerator, TimeSeriesGenerator
from src.core.streaming import ConsoleStreamer, FileStreamer, KafkaStreamer
from src.api.api_server import run_api_server

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, config_manager.get('logging.level', 'INFO')),
        format=config_manager.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
    )

def load_schema_from_file(file_path: str) -> Dict[str, Any]:
    """Load a schema from a JSON file.
    
    Args:
        file_path: Path to the JSON file.
        
    Returns:
        The loaded schema.
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading schema file: {e}")
        sys.exit(1)

def create_example_schema(schema_type: str, output_file: str):
    """Create an example schema file.
    
    Args:
        schema_type: Type of schema to create ('tabular' or 'timeseries').
        output_file: Path to save the schema.
    """
    if schema_type == 'tabular':
        schema = {
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
    elif schema_type == 'timeseries':
        schema = {
            "time_field": "timestamp",
            "time_interval_seconds": 300,  # 5 minutes
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
                    "period": 24,  # 24 hours
                    "amplitude": 0.2
                }
            }
        }
    else:
        print(f"Unknown schema type: {schema_type}")
        sys.exit(1)
    
    try:
        with open(output_file, 'w') as f:
            json.dump(schema, f, indent=2)
        print(f"Example {schema_type} schema saved to {output_file}")
    except Exception as e:
        print(f"Error saving schema file: {e}")
        sys.exit(1)

def run_single_generation(engine: SynthEngine, apply_drift: bool = False):
    """Run a single batch of data generation.
    
    Args:
        engine: The SynthEngine instance.
        apply_drift: Whether to apply drift to the generated data.
    """
    print("Generating data...")
    results = engine.run_once(apply_drift=apply_drift)
    
    if results:
        print("Data generated and streamed successfully.")
    else:
        print("No data was generated. Make sure generators and streamers are configured.")

def run_continuous_generation(engine: SynthEngine, duration: Optional[int] = None):
    """Run continuous data generation.
    
    Args:
        engine: The SynthEngine instance.
        duration: Duration to run for in seconds. If None, run indefinitely.
    """
    try:
        print("Starting continuous data generation...")
        print("Press Ctrl+C to stop.")
        
        engine.run_continuous()
        
        if duration is not None:
            time.sleep(duration)
            engine.stop()
            print(f"Stopped after {duration} seconds.")
        else:
            # Run indefinitely until interrupted
            while True:
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\nStopping data generation...")
        engine.stop()
        print("Data generation stopped.")

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Synth - Synthetic Data Generation System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Schema command
    schema_parser = subparsers.add_parser("schema", help="Schema management")
    schema_subparsers = schema_parser.add_subparsers(dest="schema_command", help="Schema command")
    
    # Create example schema
    create_schema_parser = schema_subparsers.add_parser("create-example", help="Create an example schema")
    create_schema_parser.add_argument("type", choices=["tabular", "timeseries"], help="Type of schema")
    create_schema_parser.add_argument("--output", "-o", default="schema.json", help="Output file")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run data generation")
    run_parser.add_argument("--schema", "-s", required=True, help="Schema file path")
    run_parser.add_argument("--type", "-t", choices=["tabular", "timeseries"], required=True, help="Type of data to generate")
    run_parser.add_argument("--name", "-n", default="default", help="Name for the generator")
    run_parser.add_argument("--output", "-o", default=None, help="Output file path (if not specified, outputs to console)")
    run_parser.add_argument("--format", "-f", choices=["csv", "json", "parquet"], default="csv", help="Output format")
    run_parser.add_argument("--batch-size", "-b", type=int, default=None, help="Batch size")
    run_parser.add_argument("--frequency", type=float, default=None, help="Generation frequency in seconds")
    run_parser.add_argument("--continuous", "-c", action="store_true", help="Run continuously")
    run_parser.add_argument("--duration", "-d", type=int, default=None, help="Duration to run for in seconds (only with --continuous)")
    run_parser.add_argument("--drift", action="store_true", help="Apply drift to the data")
    run_parser.add_argument("--drift-interval", type=int, default=None, help="Interval between drift events in seconds")
    run_parser.add_argument("--drift-magnitude", type=float, default=None, help="Magnitude of drift (0.0 to 1.0)")
    
    # API server command
    api_parser = subparsers.add_parser("api", help="Run API server")
    api_parser.add_argument("--host", default=None, help="API server host")
    api_parser.add_argument("--port", type=int, default=None, help="API server port")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_subparsers = config_parser.add_subparsers(dest="config_command", help="Config command")
    
    # Show config
    show_config_parser = config_subparsers.add_parser("show", help="Show current configuration")
    
    # Set config
    set_config_parser = config_subparsers.add_parser("set", help="Set configuration value")
    set_config_parser.add_argument("key", help="Configuration key (dot-separated)")
    set_config_parser.add_argument("value", help="Configuration value")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Handle commands
    if args.command == "schema":
        if args.schema_command == "create-example":
            create_example_schema(args.type, args.output)
        else:
            schema_parser.print_help()
            
    elif args.command == "run":
        # Create engine
        engine = SynthEngine()
        
        # Configure engine
        engine_config = {}
        if args.batch_size is not None:
            engine_config["batch_size"] = args.batch_size
        if args.frequency is not None:
            engine_config["generation_frequency"] = args.frequency
        if args.drift:
            engine_config["drift_enabled"] = True
        if args.drift_interval is not None:
            engine_config["drift_interval"] = args.drift_interval
        if args.drift_magnitude is not None:
            engine_config["drift_magnitude"] = args.drift_magnitude
        
        if engine_config:
            engine.configure(engine_config)
        
        # Load schema
        schema = load_schema_from_file(args.schema)
        
        # Create generator
        if args.type == "tabular":
            generator = TabularDataGenerator(args.name, schema)
        elif args.type == "timeseries":
            generator = TimeSeriesGenerator(args.name, schema)
        else:
            print(f"Unknown generator type: {args.type}")
            sys.exit(1)
        
        engine.add_generator(generator)
        
        # Create streamer
        if args.output:
            streamer = FileStreamer(
                f"{args.name}_file_streamer",
                args.output,
                format=args.format
            )
        else:
            streamer = ConsoleStreamer(f"{args.name}_console_streamer")
        
        engine.add_streamer(streamer)
        
        # Run generation
        if args.continuous:
            run_continuous_generation(engine, args.duration)
        else:
            run_single_generation(engine, args.drift)
            
        # Cleanup
        engine.close()
        
    elif args.command == "api":
        # Configure API server
        if args.host:
            config_manager.set('api.host', args.host)
        if args.port:
            config_manager.set('api.port', args.port)
        
        # Run API server
        run_api_server()
        
    elif args.command == "config":
        if args.config_command == "show":
            print(json.dumps(config_manager.config, indent=2))
        elif args.config_command == "set":
            # Try to parse the value as JSON, fall back to string
            try:
                value = json.loads(args.value)
            except:
                value = args.value
                
            config_manager.set(args.key, value)
            config_manager.save("config.yaml")
            print(f"Configuration updated: {args.key} = {value}")
        else:
            config_parser.print_help()
            
    else:
        parser.print_help()

if __name__ == "__main__":
    main()