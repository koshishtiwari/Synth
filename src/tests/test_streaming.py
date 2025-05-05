import unittest
import os
import tempfile
from io import StringIO
import json
import pandas as pd
from unittest.mock import patch, MagicMock

from src.core.streaming import ConsoleStreamer, FileStreamer, StreamingManager

class TestConsoleStreamer(unittest.TestCase):
    """Tests for the ConsoleStreamer class."""
    
    def setUp(self):
        """Set up a test streamer."""
        self.streamer = ConsoleStreamer("test_console")
        
    @patch('sys.stdout', new_callable=StringIO)
    def test_send_dataframe(self, mock_stdout):
        """Test sending a DataFrame to the console."""
        # Create a test dataframe
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10.0, 20.0, 30.0],
            'category': ['A', 'B', 'A']
        })
        
        # Send to console
        result = self.streamer.send(df)
        
        # Check result and output
        self.assertTrue(result)
        output = mock_stdout.getvalue()
        self.assertIn("test_console Output", output)
        self.assertIn("id", output)
        self.assertIn("value", output)
        self.assertIn("category", output)
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_send_dict(self, mock_stdout):
        """Test sending a dictionary to the console."""
        # Create a test dict
        data = {
            'id': 1,
            'value': 10.0,
            'category': 'A'
        }
        
        # Send to console
        result = self.streamer.send(data)
        
        # Check result and output
        self.assertTrue(result)
        output = mock_stdout.getvalue()
        self.assertIn("test_console Output", output)
        self.assertIn("id", output)
        self.assertIn("value", output)
        self.assertIn("category", output)

class TestFileStreamer(unittest.TestCase):
    """Tests for the FileStreamer class."""
    
    def setUp(self):
        """Set up a test streamer."""
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_file = os.path.join(self.temp_dir.name, "test_output.csv")
        self.streamer = FileStreamer("test_file", self.test_file, format="csv")
        
    def tearDown(self):
        """Clean up resources."""
        self.temp_dir.cleanup()
        
    def test_send_dataframe_csv(self):
        """Test sending a DataFrame to a CSV file."""
        # Create a test dataframe
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10.0, 20.0, 30.0],
            'category': ['A', 'B', 'A']
        })
        
        # Send to file
        result = self.streamer.send(df)
        
        # Check result and file content
        self.assertTrue(result)
        self.assertTrue(os.path.exists(self.test_file))
        
        # Read back and verify
        read_df = pd.read_csv(self.test_file)
        self.assertEqual(len(read_df), 3)
        self.assertTrue('id' in read_df.columns)
        self.assertTrue('value' in read_df.columns)
        self.assertTrue('category' in read_df.columns)
    
    def test_append_mode(self):
        """Test appending to a file."""
        # Create a test dataframe
        df1 = pd.DataFrame({
            'id': [1, 2],
            'value': [10.0, 20.0],
            'category': ['A', 'B']
        })
        
        df2 = pd.DataFrame({
            'id': [3, 4],
            'value': [30.0, 40.0],
            'category': ['C', 'D']
        })
        
        # Send first batch
        self.streamer.send(df1)
        
        # Send second batch
        self.streamer.send(df2)
        
        # Read back and verify
        read_df = pd.read_csv(self.test_file)
        self.assertEqual(len(read_df), 4)

class TestStreamingManager(unittest.TestCase):
    """Tests for the StreamingManager class."""
    
    def setUp(self):
        """Set up a test manager with mock streamers."""
        self.manager = StreamingManager()
        
        # Create mock streamers
        self.mock_streamer1 = MagicMock()
        self.mock_streamer1.name = "streamer1"
        self.mock_streamer1.send.return_value = True
        
        self.mock_streamer2 = MagicMock()
        self.mock_streamer2.name = "streamer2"
        self.mock_streamer2.send.return_value = True
        
        # Add to manager
        self.manager.add_streamer(self.mock_streamer1)
        self.manager.add_streamer(self.mock_streamer2)
        
    def test_add_remove_streamer(self):
        """Test adding and removing streamers."""
        # Check initial state
        self.assertEqual(len(self.manager.streamers), 2)
        self.assertTrue("streamer1" in self.manager.streamers)
        self.assertTrue("streamer2" in self.manager.streamers)
        
        # Remove a streamer
        self.manager.remove_streamer("streamer1")
        
        # Check updated state
        self.assertEqual(len(self.manager.streamers), 1)
        self.assertFalse("streamer1" in self.manager.streamers)
        self.assertTrue("streamer2" in self.manager.streamers)
        
    def test_stream(self):
        """Test streaming data to all streamers."""
        # Create test data
        data = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10.0, 20.0, 30.0]
        })
        
        # Stream the data
        results = self.manager.stream(data)
        
        # Check results
        self.assertEqual(len(results), 2)
        self.assertTrue(results["streamer1"])
        self.assertTrue(results["streamer2"])
        
        # Verify both streamers were called with the data
        self.mock_streamer1.send.assert_called_once()
        self.mock_streamer2.send.assert_called_once()
    
    def test_close_all(self):
        """Test closing all streamers."""
        # Close all streamers
        self.manager.close_all()
        
        # Verify close was called on both streamers
        self.mock_streamer1.close.assert_called_once()
        self.mock_streamer2.close.assert_called_once()
        
        # Check streamers dict is empty
        self.assertEqual(len(self.manager.streamers), 0)
        
if __name__ == "__main__":
    unittest.main()