import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Set up metrics for Prometheus monitoring
METRICS = {
    'batches_generated': Counter('synth_batches_generated_total', 'Total number of batches generated', ['generator']),
    'records_generated': Counter('synth_records_generated_total', 'Total number of records generated', ['generator']),
    'drift_events': Counter('synth_drift_events_total', 'Total number of drift events', ['generator']),
    'generation_time': Histogram('synth_generation_time_seconds', 'Time taken to generate a batch', ['generator']),
    'stream_time': Histogram('synth_stream_time_seconds', 'Time taken to stream a batch', ['streamer']),
    'active_generators': Gauge('synth_active_generators', 'Number of active generators'),
    'active_streamers': Gauge('synth_active_streamers', 'Number of active streamers'),
}

def start_metrics_server(port: int = 8001) -> None:
    """Start Prometheus metrics server.
    
    Args:
        port: Port to run the server on.
    """
    start_http_server(port)
    logging.info(f"Started metrics server on port {port}")

def record_metric(name: str, value: float = 1, labels: Dict[str, str] = None) -> None:
    """Record a metric.
    
    Args:
        name: Name of the metric.
        value: Value to record.
        labels: Labels for the metric.
    """
    if name not in METRICS:
        logging.warning(f"Unknown metric: {name}")
        return
        
    metric = METRICS[name]
    
    if hasattr(metric, 'inc'):
        if labels:
            metric.labels(**labels).inc(value)
        else:
            metric.inc(value)
    elif hasattr(metric, 'set'):
        if labels:
            metric.labels(**labels).set(value)
        else:
            metric.set(value)
    elif hasattr(metric, 'observe'):
        if labels:
            metric.labels(**labels).observe(value)
        else:
            metric.observe(value)

def validate_schema(schema: Dict[str, Any]) -> List[str]:
    """Validate a data schema.
    
    Args:
        schema: Schema to validate.
        
    Returns:
        List of validation errors or empty list if valid.
    """
    errors = []
    
    # Check each field in the schema
    for field_name, field_def in schema.items():
        if not isinstance(field_def, dict):
            errors.append(f"Field '{field_name}' must be a dictionary")
            continue
            
        # Check if type is specified for regular fields
        if field_name not in ['time_field', 'time_interval_seconds', 'trends', 'seasonality']:
            if 'type' not in field_def:
                errors.append(f"Field '{field_name}' is missing required property 'type'")
            else:
                field_type = field_def['type']
                
                # Validate field type specific properties
                if field_type == 'float':
                    # Check that mean and std are valid
                    if 'mean' in field_def and not isinstance(field_def['mean'], (int, float)):
                        errors.append(f"Field '{field_name}': 'mean' must be a number")
                    if 'std' in field_def and not isinstance(field_def['std'], (int, float)):
                        errors.append(f"Field '{field_name}': 'std' must be a number")
                    if 'std' in field_def and field_def['std'] < 0:
                        errors.append(f"Field '{field_name}': 'std' must be non-negative")
                        
                elif field_type == 'int':
                    # Check that min and max are valid
                    if 'min' in field_def and not isinstance(field_def['min'], int):
                        errors.append(f"Field '{field_name}': 'min' must be an integer")
                    if 'max' in field_def and not isinstance(field_def['max'], int):
                        errors.append(f"Field '{field_name}': 'max' must be an integer")
                    if 'min' in field_def and 'max' in field_def and field_def['min'] > field_def['max']:
                        errors.append(f"Field '{field_name}': 'min' must be less than or equal to 'max'")
                        
                elif field_type == 'category':
                    # Check that categories is a list and probabilities sum to 1
                    if 'categories' not in field_def:
                        errors.append(f"Field '{field_name}': 'categories' is required for category type")
                    elif not isinstance(field_def['categories'], list):
                        errors.append(f"Field '{field_name}': 'categories' must be a list")
                    elif len(field_def['categories']) == 0:
                        errors.append(f"Field '{field_name}': 'categories' must not be empty")
                        
                    if 'probabilities' in field_def:
                        if not isinstance(field_def['probabilities'], list):
                            errors.append(f"Field '{field_name}': 'probabilities' must be a list")
                        elif len(field_def['probabilities']) != len(field_def.get('categories', [])):
                            errors.append(f"Field '{field_name}': 'probabilities' length must match 'categories' length")
                        elif any(not isinstance(p, (int, float)) for p in field_def['probabilities']):
                            errors.append(f"Field '{field_name}': 'probabilities' must contain only numbers")
                        elif abs(sum(field_def['probabilities']) - 1.0) > 0.0001:
                            errors.append(f"Field '{field_name}': 'probabilities' must sum to 1.0")
                
                elif field_type == 'datetime':
                    # Validate date format if provided
                    if 'start_date' in field_def and not isinstance(field_def['start_date'], (str, datetime)):
                        errors.append(f"Field '{field_name}': 'start_date' must be a date string or datetime")
                    if 'end_date' in field_def and not isinstance(field_def['end_date'], (str, datetime)):
                        errors.append(f"Field '{field_name}': 'end_date' must be a date string or datetime")
    
    # Check time series specific fields if present
    if 'time_field' in schema and not isinstance(schema['time_field'], str):
        errors.append("'time_field' must be a string")
        
    if 'time_interval_seconds' in schema and not isinstance(schema['time_interval_seconds'], (int, float)):
        errors.append("'time_interval_seconds' must be a number")
    
    return errors

def export_data_sample(data: pd.DataFrame, output_path: str, format: str = 'csv') -> None:
    """Export a sample of data to a file.
    
    Args:
        data: DataFrame to export.
        output_path: Path to save the file.
        format: Format to save as ('csv', 'json', or 'parquet').
    """
    format = format.lower()
    
    if format == 'csv':
        data.to_csv(output_path, index=False)
    elif format == 'json':
        data.to_json(output_path, orient='records', lines=True)
    elif format == 'parquet':
        try:
            data.to_parquet(output_path, index=False)
        except ImportError:
            logging.error("pyarrow package is required for Parquet support")
            raise
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logging.info(f"Exported {len(data)} records to {output_path}")

def analyze_data(data: pd.DataFrame) -> Dict[str, Any]:
    """Analyze a DataFrame and return statistics.
    
    Args:
        data: DataFrame to analyze.
        
    Returns:
        Dictionary with analysis results.
    """
    result = {
        'record_count': len(data),
        'column_count': len(data.columns),
        'columns': {},
        'memory_usage': data.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
    }
    
    for column in data.columns:
        col_data = data[column]
        col_type = str(col_data.dtype)
        
        col_stats = {
            'dtype': col_type,
            'null_count': col_data.isna().sum(),
            'unique_count': col_data.nunique(),
        }
        
        # Add type-specific stats
        if np.issubdtype(col_data.dtype, np.number):
            col_stats.update({
                'min': col_data.min(),
                'max': col_data.max(),
                'mean': col_data.mean(),
                'std': col_data.std(),
                'median': col_data.median(),
            })
        elif col_data.dtype == 'object' or col_data.dtype == 'category':
            try:
                if col_data.nunique() <= 10:  # Only show value counts for categorical-like columns
                    col_stats['value_counts'] = col_data.value_counts().to_dict()
            except:
                pass
        
        result['columns'][column] = col_stats
    
    return result