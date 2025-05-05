import unittest
import pandas as pd
import numpy as np
from src.models.data_generators import TabularDataGenerator, TimeSeriesGenerator

class TestTabularDataGenerator(unittest.TestCase):
    """Tests for the TabularDataGenerator class."""
    
    def setUp(self):
        """Set up a test generator."""
        self.schema = {
            "id": {
                "type": "int",
                "min": 1000,
                "max": 9999
            },
            "value": {
                "type": "float",
                "mean": 100,
                "std": 15,
                "min": 0
            },
            "category": {
                "type": "category",
                "categories": ["A", "B", "C"],
                "probabilities": [0.5, 0.3, 0.2]
            }
        }
        self.generator = TabularDataGenerator("test_tabular", self.schema)
        
    def test_generate_batch(self):
        """Test generating a batch of data."""
        batch_size = 100
        df = self.generator.generate_batch(batch_size)
        
        # Check the dataframe has the right shape
        self.assertEqual(len(df), batch_size)
        self.assertEqual(set(df.columns), set(self.schema.keys()))
        
        # Check id column is within range
        self.assertTrue(df["id"].min() >= 1000)
        self.assertTrue(df["id"].max() <= 9999)
        
        # Check value column has appropriate statistics
        self.assertTrue(df["value"].min() >= 0)
        
        # Check category column has only expected values
        self.assertTrue(set(df["category"].unique()).issubset(set(["A", "B", "C"])))
        
    def test_apply_drift(self):
        """Test applying drift to generated data."""
        batch_size = 100
        df = self.generator.generate_batch(batch_size)
        
        # Apply drift
        drifted_df = self.generator.apply_drift(df, 0.5)
        
        # Check that the drifted dataframe is different but has the same structure
        self.assertEqual(len(drifted_df), len(df))
        self.assertEqual(set(drifted_df.columns), set(df.columns))
        
        # Check that some values have changed
        self.assertFalse(np.array_equal(df["value"].values, drifted_df["value"].values))

class TestTimeSeriesGenerator(unittest.TestCase):
    """Tests for the TimeSeriesGenerator class."""
    
    def setUp(self):
        """Set up a test generator."""
        self.schema = {
            "time_field": "timestamp",
            "time_interval_seconds": 300,  # 5 minutes
            "value": {
                "type": "float",
                "mean": 100,
                "std": 15,
                "min": 0
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
        self.generator = TimeSeriesGenerator("test_timeseries", self.schema)
        
    def test_generate_batch(self):
        """Test generating a batch of time series data."""
        batch_size = 100
        df = self.generator.generate_batch(batch_size)
        
        # Check the dataframe has the right shape
        self.assertEqual(len(df), batch_size)
        self.assertTrue("timestamp" in df.columns)
        self.assertTrue("value" in df.columns)
        
        # Check timestamps are sequential
        timestamps = pd.to_datetime(df["timestamp"])
        self.assertTrue(timestamps.is_monotonic_increasing)
        
if __name__ == "__main__":
    unittest.main()