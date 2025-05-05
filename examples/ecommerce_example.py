#!/usr/bin/env python
"""
Example: E-commerce Data Generator

This example shows how to use the Synth system to generate realistic e-commerce data,
including customer profiles and transaction time series with seasonal patterns.
"""

import os
import json
import logging
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd

from src.config.config_manager import config_manager
from src.core.engine import SynthEngine
from src.models.data_generators import TabularDataGenerator, TimeSeriesGenerator
from src.core.streaming import ConsoleStreamer, FileStreamer
from src.utils.monitoring import analyze_data, validate_schema, export_data_sample

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ecommerce_example")

# Create output directory
output_dir = "ecommerce_data"
os.makedirs(output_dir, exist_ok=True)

def main():
    """Run the e-commerce data generation example."""
    logger.info("Starting e-commerce data generation example")
    
    # Initialize the engine
    engine = SynthEngine()
    
    # Configure batch size and frequency
    engine.configure({
        "batch_size": 100,  # Generate 100 records at a time
        "generation_frequency": 0.1,  # Generate every 100ms for demo purposes
        "drift_enabled": True,
        "drift_interval": 10,  # Apply drift every 10 seconds
        "drift_magnitude": 0.2  # Mild drift
    })
    
    # 1. Create customer profiles generator
    customer_schema = {
        "customer_id": {
            "type": "int",
            "min": 10000,
            "max": 99999
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
            "max": 85
        },
        "gender": {
            "type": "category",
            "categories": ["M", "F", "Other"],
            "probabilities": [0.48, 0.48, 0.04]
        },
        "location": {
            "type": "category",
            "categories": ["North", "South", "East", "West", "Central"],
            "probabilities": [0.2, 0.2, 0.2, 0.2, 0.2]
        },
        "signup_date": {
            "type": "datetime",
            "start_date": "2020-01-01",
            "end_date": "2023-12-31"
        },
        "is_premium": {
            "type": "category",
            "categories": [True, False],
            "probabilities": [0.25, 0.75]
        },
        "lifetime_value": {
            "type": "float",
            "mean": 500,
            "std": 300,
            "min": 0
        }
    }
    
    # Validate the schema
    errors = validate_schema(customer_schema)
    if errors:
        for error in errors:
            logger.error(f"Schema validation error: {error}")
        return
    
    # Create and add the customer generator
    customer_generator = TabularDataGenerator("customer_profiles", customer_schema)
    engine.add_generator(customer_generator)
    
    # 2. Create transaction time series generator
    transaction_schema = {
        "time_field": "timestamp",
        "time_interval_seconds": 300,  # 5 minutes between data points
        "transaction_id": {
            "type": "int",
            "min": 1000000,
            "max": 9999999
        },
        "customer_id": {
            "type": "int",
            "min": 10000,
            "max": 99999
        },
        "product_id": {
            "type": "int",
            "min": 1,
            "max": 1000
        },
        "quantity": {
            "type": "int",
            "min": 1,
            "max": 10
        },
        "price": {
            "type": "float",
            "mean": 50,
            "std": 30,
            "min": 5,
            "max": 500
        },
        "discount": {
            "type": "float",
            "mean": 0.1,
            "std": 0.1,
            "min": 0,
            "max": 0.5
        },
        "payment_method": {
            "type": "category",
            "categories": ["credit_card", "debit_card", "paypal", "apple_pay", "google_pay"],
            "probabilities": [0.4, 0.3, 0.15, 0.1, 0.05]
        },
        # Add seasonal patterns to transaction data
        "trends": {
            "price": {
                "type": "linear",
                "strength": 0.05  # Slight upward trend in prices over time
            }
        },
        "seasonality": {
            "quantity": {
                "period": 24,  # Daily seasonality (24 hours)
                "amplitude": 0.3  # Higher quantities during peak hours
            },
            "discount": {
                "period": 168,  # Weekly seasonality (168 hours)
                "amplitude": 0.2  # Higher discounts on weekends
            }
        }
    }
    
    # Validate the schema
    errors = validate_schema(transaction_schema)
    if errors:
        for error in errors:
            logger.error(f"Schema validation error: {error}")
        return
    
    # Create and add the transaction generator
    transaction_generator = TimeSeriesGenerator("transactions", transaction_schema)
    engine.add_generator(transaction_generator)
    
    # 3. Set up streamers for different outputs
    
    # Console streamer for immediate feedback
    console_streamer = ConsoleStreamer("console_output")
    engine.add_streamer(console_streamer)
    
    # File streamers for persistent storage
    customer_file_streamer = FileStreamer(
        "customer_file",
        os.path.join(output_dir, "customers.csv"),
        format="csv"
    )
    engine.add_streamer(customer_file_streamer)
    
    transaction_file_streamer = FileStreamer(
        "transaction_file",
        os.path.join(output_dir, "transactions.csv"),
        format="csv"
    )
    engine.add_streamer(transaction_file_streamer)
    
    # Also save as JSON for API simulation
    transaction_json_streamer = FileStreamer(
        "transaction_json",
        os.path.join(output_dir, "transactions.jsonl"),
        format="json"
    )
    engine.add_streamer(transaction_json_streamer)
    
    # 4. Generate some data in single-run mode to analyze
    logger.info("Generating initial data samples for analysis...")
    engine.run_once()
    
    # Get generated data for analysis
    customer_data = engine.generate_batch("customer_profiles")["customer_profiles"]
    transaction_data = engine.generate_batch("transactions")["transactions"]
    
    # Save samples for inspection
    customer_data.to_csv(os.path.join(output_dir, "customer_sample.csv"), index=False)
    transaction_data.to_csv(os.path.join(output_dir, "transaction_sample.csv"), index=False)
    
    # Analyze the data
    logger.info("Analyzing generated data...")
    customer_analysis = analyze_data(customer_data)
    transaction_analysis = analyze_data(transaction_data)
    
    # Save analysis results
    with open(os.path.join(output_dir, "customer_analysis.json"), "w") as f:
        json.dump(customer_analysis, f, indent=2, default=str)
    
    with open(os.path.join(output_dir, "transaction_analysis.json"), "w") as f:
        json.dump(transaction_analysis, f, indent=2, default=str)
    
    # Generate visualization of the data
    visualize_data(customer_data, transaction_data, output_dir)
    
    # 5. Run continuous generation for a short time
    logger.info("Starting continuous data generation (10 seconds)...")
    engine.run_continuous()
    
    try:
        # Run for 10 seconds
        time.sleep(10)
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
    finally:
        engine.stop()
        logger.info("Continuous generation stopped")
    
    # Get statistics
    stats = engine.get_stats()
    logger.info(f"Generation statistics: {stats}")
    
    # Write stats to file
    with open(os.path.join(output_dir, "generation_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, default=str)
    
    logger.info(f"Example complete. Data saved to '{output_dir}' directory")

def visualize_data(customer_data, transaction_data, output_dir):
    """Create visualizations of the generated data.
    
    Args:
        customer_data: Customer profile data
        transaction_data: Transaction data
        output_dir: Directory to save visualizations
    """
    plt.figure(figsize=(15, 10))
    
    # Customer age distribution
    plt.subplot(2, 2, 1)
    customer_data['age'].hist(bins=20)
    plt.title('Customer Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Count')
    
    # Customer lifetime value by age
    plt.subplot(2, 2, 2)
    plt.scatter(customer_data['age'], customer_data['lifetime_value'], alpha=0.5)
    plt.title('Customer Lifetime Value by Age')
    plt.xlabel('Age')
    plt.ylabel('Lifetime Value ($)')
    
    # Transaction prices
    plt.subplot(2, 2, 3)
    transaction_data['price'].hist(bins=20)
    plt.title('Transaction Price Distribution')
    plt.xlabel('Price ($)')
    plt.ylabel('Count')
    
    # Transactions by payment method
    plt.subplot(2, 2, 4)
    transaction_data['payment_method'].value_counts().plot(kind='bar')
    plt.title('Transactions by Payment Method')
    plt.xlabel('Payment Method')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "data_visualization.png"))
    plt.close()
    
    # Time series visualization if we have enough time points
    if len(transaction_data) > 10:
        plt.figure(figsize=(15, 6))
        
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(transaction_data['timestamp']):
            transaction_data['timestamp'] = pd.to_datetime(transaction_data['timestamp'])
        
        # Sort by timestamp
        transaction_data = transaction_data.sort_values('timestamp')
        
        # Plot quantity over time
        plt.subplot(1, 2, 1)
        plt.plot(transaction_data['timestamp'], transaction_data['quantity'])
        plt.title('Quantity Over Time')
        plt.xlabel('Time')
        plt.ylabel('Quantity')
        plt.xticks(rotation=45)
        
        # Plot price over time
        plt.subplot(1, 2, 2)
        plt.plot(transaction_data['timestamp'], transaction_data['price'])
        plt.title('Price Over Time')
        plt.xlabel('Time')
        plt.ylabel('Price ($)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "time_series_visualization.png"))
        plt.close()

if __name__ == "__main__":
    main()