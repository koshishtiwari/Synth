import unittest
import threading
import time
from unittest.mock import MagicMock, patch
import pandas as pd

from src.core.engine import SynthEngine
from src.core.data_generator import BaseDataGenerator
from src.core.streaming import BaseStreamer

class MockDataGenerator(BaseDataGenerator):
    """Mock data generator for testing."""
    
    def __init__(self, name, schema):
        super().__init__(name, schema)
        self.generate_batch_called = 0
        self.apply_drift_called = 0
        
    def generate_batch(self, size):
        """Generate a mock batch of data."""
        self.generate_batch_called += 1
        return pd.DataFrame({
            'id': range(size),
            'value': [1.0] * size,
            'category': ['A'] * size
        })
        
    def apply_drift(self, data, magnitude):
        """Apply mock drift to data."""
        self.apply_drift_called += 1
        drifted = data.copy()
        drifted['value'] = data['value'] + magnitude
        return drifted

class MockStreamer(BaseStreamer):
    """Mock streamer for testing."""
    
    def __init__(self, name):
        super().__init__(name)
        self.send_called = 0
        self.close_called = 0
        self.last_data = None
        
    def send(self, data):
        """Mock sending data."""
        self.send_called += 1
        self.last_data = data
        return True
        
    def close(self):
        """Mock closing the streamer."""
        self.close_called += 1

class TestSynthEngine(unittest.TestCase):
    """Tests for the SynthEngine class."""
    
    def setUp(self):
        """Set up a test engine with mock components."""
        self.engine = SynthEngine()
        
        # Add mock generator
        self.generator = MockDataGenerator("mock_gen", {})
        self.engine.add_generator(self.generator)
        
        # Add mock streamer
        self.streamer = MockStreamer("mock_stream")
        self.engine.add_streamer(self.streamer)
        
    def test_add_remove_generator(self):
        """Test adding and removing generators."""
        # Check initial state
        self.assertEqual(len(self.engine.generators), 1)
        self.assertTrue("mock_gen" in self.engine.generators)
        
        # Add another generator
        gen2 = MockDataGenerator("mock_gen2", {})
        self.engine.add_generator(gen2)
        
        # Check updated state
        self.assertEqual(len(self.engine.generators), 2)
        self.assertTrue("mock_gen2" in self.engine.generators)
        
        # Remove a generator
        self.engine.remove_generator("mock_gen")
        
        # Check final state
        self.assertEqual(len(self.engine.generators), 1)
        self.assertFalse("mock_gen" in self.engine.generators)
        self.assertTrue("mock_gen2" in self.engine.generators)
        
    def test_generate_batch(self):
        """Test generating a batch of data."""
        # Generate batch
        result = self.engine.generate_batch()
        
        # Check result
        self.assertEqual(len(result), 1)
        self.assertTrue("mock_gen" in result)
        self.assertEqual(self.generator.generate_batch_called, 1)
        
        # Check with drift
        result = self.engine.generate_batch(apply_drift=True)
        self.assertEqual(self.generator.apply_drift_called, 1)
        
    def test_run_once(self):
        """Test running the engine once."""
        # Run once
        self.engine.run_once()
        
        # Check generator and streamer were called
        self.assertEqual(self.generator.generate_batch_called, 1)
        self.assertEqual(self.streamer.send_called, 1)
        
        # Check with drift
        self.engine.run_once(apply_drift=True)
        self.assertEqual(self.generator.apply_drift_called, 1)
        
    def test_run_continuous(self):
        """Test running the engine continuously."""
        # Start continuous run
        self.engine.generation_frequency = 0.1  # Make it run fast for testing
        self.engine.run_continuous()
        
        # Check it's running
        self.assertTrue(self.engine.running)
        self.assertIsNotNone(self.engine.thread)
        
        # Let it run for a short time
        time.sleep(0.3)
        
        # Stop it
        self.engine.stop()
        
        # Check it's stopped
        self.assertFalse(self.engine.running)
        
        # Verify it generated multiple batches
        self.assertGreater(self.generator.generate_batch_called, 1)
        self.assertGreater(self.streamer.send_called, 1)
        
    def test_configure(self):
        """Test configuring the engine."""
        # Initial values
        initial_batch_size = self.engine.batch_size
        initial_frequency = self.engine.generation_frequency
        
        # Configure with new values
        self.engine.configure({
            "batch_size": 500,
            "generation_frequency": 2.0,
            "drift_enabled": True,
            "drift_interval": 120,
            "drift_magnitude": 0.3
        })
        
        # Check configuration took effect
        self.assertEqual(self.engine.batch_size, 500)
        self.assertEqual(self.engine.generation_frequency, 2.0)
        self.assertTrue(self.engine.drift_enabled)
        self.assertEqual(self.engine.drift_interval, 120)
        self.assertEqual(self.engine.drift_magnitude, 0.3)
        
if __name__ == "__main__":
    unittest.main()