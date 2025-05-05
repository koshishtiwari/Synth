import json
import aiohttp
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta

from src.data_sources.base_source import BaseDataSource

class MarketDataSource(BaseDataSource):
    """Base class for market data sources."""
    
    async def preprocess_data(self, data: Any) -> pd.DataFrame:
        """Preprocess raw API data into a pandas DataFrame.
        
        Args:
            data: Raw data from the API.
            
        Returns:
            Processed DataFrame.
        """
        # Implement in subclasses
        raise NotImplementedError("Data preprocessing not implemented for this data source")

class AlphaVantageSource(MarketDataSource):
    """Data source for Alpha Vantage API.
    
    Provides access to stock market data, forex, and cryptocurrencies.
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """Initialize the Alpha Vantage data source.
        
        Args:
            name: Name of the data source.
            config: Configuration parameters for the data source.
                - api_key: Alpha Vantage API key.
                - base_url: Base URL for the API (optional).
        """
        super().__init__(name, config)
        
        # Check for API key
        self.api_key = self.config.get('api_key')
        if not self.api_key:
            self.api_key = config_manager.get('alpha_vantage.api_key')
            
        if not self.api_key:
            self.logger.warning("No Alpha Vantage API key provided. Data will be limited or unavailable.")
            
        self.base_url = self.config.get('base_url', 'https://www.alphavantage.co/query')
        
    async def fetch_data(self, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Fetch data from Alpha Vantage API.
        
        Args:
            params: Parameters for the API request.
                - function: API function to call (e.g., TIME_SERIES_INTRADAY).
                - symbol: Stock symbol (e.g., MSFT).
                - interval: Time interval (e.g., 1min, 5min, 15min, 30min, 60min).
                - adjusted: Whether to return adjusted data (True/False).
                - outputsize: compact or full.
                - datatype: json or csv.
                
        Returns:
            DataFrame with the fetched data.
        """
        params = params or {}
        
        # Build request parameters
        request_params = {
            'apikey': self.api_key,
            'function': params.get('function', 'TIME_SERIES_INTRADAY'),
            'symbol': params.get('symbol', 'MSFT'),
            'interval': params.get('interval', '5min'),
            'adjusted': 'true' if params.get('adjusted', True) else 'false',
            'outputsize': params.get('outputsize', 'compact'),
            'datatype': params.get('datatype', 'json')
        }
        
        # Make API request
        async with aiohttp.ClientSession() as session:
            async with session.get(self.base_url, params=request_params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self.logger.error(f"Alpha Vantage API error: {response.status}, {error_text}")
                    raise Exception(f"API request failed with status {response.status}: {error_text}")
                
                data = await response.json()
                
                # Check for API error messages
                if 'Error Message' in data:
                    self.logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                    raise Exception(f"API error: {data['Error Message']}")
                    
                # Check for API rate limit messages
                if 'Note' in data and 'API call frequency' in data['Note']:
                    self.logger.warning(f"Alpha Vantage API rate limit warning: {data['Note']}")
                
                return await self.preprocess_data(data)
    
    async def preprocess_data(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess Alpha Vantage data into a pandas DataFrame.
        
        Args:
            data: Raw data from the Alpha Vantage API.
            
        Returns:
            Processed DataFrame.
        """
        # Extract metadata
        metadata = data.get('Meta Data', {})
        
        # Determine which time series key to use based on the data
        time_series_key = None
        for key in data.keys():
            if 'Time Series' in key:
                time_series_key = key
                break
                
        if not time_series_key:
            self.logger.error("No time series data found in Alpha Vantage response")
            return pd.DataFrame()
            
        # Extract time series data
        time_series = data[time_series_key]
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        
        # Rename columns to remove number prefixes
        df.columns = [col.split('. ')[1] if '. ' in col else col for col in df.columns]
        
        # Convert columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Convert index to datetime
        df.index = pd.to_datetime(df.index)
        
        # Sort by timestamp
        df = df.sort_index()
        
        return df
    
    async def get_historical_data(
        self, 
        start_date: Union[str, datetime], 
        end_date: Union[str, datetime] = None,
        params: Dict[str, Any] = None
    ) -> pd.DataFrame:
        """Get historical data from Alpha Vantage.
        
        Args:
            start_date: Start date for historical data.
            end_date: End date for historical data (defaults to current time).
            params: Additional parameters for the API request.
            
        Returns:
            DataFrame with historical data.
        """
        params = params or {}
        
        # Convert dates to string format if needed
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
            
        if end_date is None:
            end_date = datetime.now()
            
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
            
        # Determine the appropriate function based on the required time range
        time_diff = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        
        if time_diff <= 5:  # For very short ranges, use intraday
            function = 'TIME_SERIES_INTRADAY'
            interval = params.get('interval', '5min')
        elif time_diff <= 100:  # For medium ranges, use daily
            function = 'TIME_SERIES_DAILY'
            interval = None
        else:  # For long ranges, use weekly
            function = 'TIME_SERIES_WEEKLY'
            interval = None
            
        # Update params with the appropriate function
        request_params = params.copy()
        request_params['function'] = function
        request_params['outputsize'] = 'full'  # Get as much data as possible
        
        if interval:
            request_params['interval'] = interval
            
        # Fetch data
        data, metadata = await self.get_data(request_params)
        
        # Filter to the requested date range
        if not data.empty:
            start_date_dt = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date)
            
            data = data[(data.index >= start_date_dt) & (data.index <= end_date_dt)]
            
        return data

class YahooFinanceSource(MarketDataSource):
    """Data source for Yahoo Finance (unofficial API).
    
    Provides access to stock market data.
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """Initialize the Yahoo Finance data source.
        
        Args:
            name: Name of the data source.
            config: Configuration parameters for the data source.
                - base_url: Base URL for the API (optional).
        """
        super().__init__(name, config)
        self.base_url = self.config.get('base_url', 'https://query1.finance.yahoo.com/v8/finance/chart/')
        
    async def fetch_data(self, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Fetch data from Yahoo Finance API.
        
        Args:
            params: Parameters for the API request.
                - symbol: Stock symbol (e.g., MSFT).
                - interval: Time interval (e.g., 1m, 5m, 15m, 30m, 60m, 1d, 1wk, 1mo).
                - range: Data range (e.g., 1d, 5d, 1mo, 3mo, 6mo, 1y, 5y, max).
                
        Returns:
            DataFrame with the fetched data.
        """
        params = params or {}
        
        symbol = params.get('symbol', 'MSFT')
        interval = params.get('interval', '5m')
        data_range = params.get('range', '1d')
        
        # Build URL
        url = f"{self.base_url}{symbol}"
        
        # Build request parameters
        request_params = {
            'interval': interval,
            'range': data_range
        }
        
        # Make API request
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=request_params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self.logger.error(f"Yahoo Finance API error: {response.status}, {error_text}")
                    raise Exception(f"API request failed with status {response.status}: {error_text}")
                
                data = await response.json()
                return await self.preprocess_data(data)
    
    async def preprocess_data(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess Yahoo Finance data into a pandas DataFrame.
        
        Args:
            data: Raw data from the Yahoo Finance API.
            
        Returns:
            Processed DataFrame.
        """
        # Extract chart data
        chart = data.get('chart', {})
        result = chart.get('result', [])
        
        if not result:
            self.logger.error("No data found in Yahoo Finance response")
            return pd.DataFrame()
            
        # First result contains the data
        result = result[0]
        
        # Extract timestamps and indicators
        timestamps = result.get('timestamp', [])
        indicators = result.get('indicators', {})
        quote = indicators.get('quote', [{}])[0]
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': quote.get('open', []),
            'high': quote.get('high', []),
            'low': quote.get('low', []),
            'close': quote.get('close', []),
            'volume': quote.get('volume', [])
        })
        
        # Add timestamps as index
        if timestamps:
            df.index = pd.to_datetime(timestamps, unit='s')
            
        # Sort by timestamp
        df = df.sort_index()
        
        return df
    
    async def get_historical_data(
        self, 
        start_date: Union[str, datetime], 
        end_date: Union[str, datetime] = None,
        params: Dict[str, Any] = None
    ) -> pd.DataFrame:
        """Get historical data from Yahoo Finance.
        
        Args:
            start_date: Start date for historical data.
            end_date: End date for historical data (defaults to current time).
            params: Additional parameters for the API request.
            
        Returns:
            DataFrame with historical data.
        """
        params = params or {}
        
        # Convert dates to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
            
        if end_date is None:
            end_date = datetime.now()
            
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        # Calculate time difference and determine appropriate range parameter
        time_diff = (end_date - start_date).days
        
        if time_diff <= 5:
            data_range = '5d'
            interval = params.get('interval', '5m')
        elif time_diff <= 30:
            data_range = '1mo'
            interval = params.get('interval', '30m')
        elif time_diff <= 90:
            data_range = '3mo'
            interval = params.get('interval', '1h')
        elif time_diff <= 180:
            data_range = '6mo'
            interval = params.get('interval', '1d')
        elif time_diff <= 365:
            data_range = '1y'
            interval = params.get('interval', '1d')
        else:
            data_range = 'max'
            interval = params.get('interval', '1wk')
            
        # Update params
        request_params = params.copy()
        request_params['range'] = data_range
        request_params['interval'] = interval
        
        # Fetch data
        data, metadata = await self.get_data(request_params)
        
        # Filter to the requested date range
        if not data.empty:
            data = data[(data.index >= start_date) & (data.index <= end_date)]
            
        return data

class CryptoCompareSource(MarketDataSource):
    """Data source for CryptoCompare API.
    
    Provides access to cryptocurrency market data.
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """Initialize the CryptoCompare data source.
        
        Args:
            name: Name of the data source.
            config: Configuration parameters for the data source.
                - api_key: CryptoCompare API key (optional).
                - base_url: Base URL for the API (optional).
        """
        super().__init__(name, config)
        
        # Check for API key
        self.api_key = self.config.get('api_key')
        if not self.api_key:
            self.api_key = config_manager.get('cryptocompare.api_key')
            
        self.base_url = self.config.get('base_url', 'https://min-api.cryptocompare.com/data/')
        
    async def fetch_data(self, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Fetch data from CryptoCompare API.
        
        Args:
            params: Parameters for the API request.
                - endpoint: API endpoint (e.g., histominute, histohour, histoday).
                - fsym: From symbol (e.g., BTC).
                - tsym: To symbol (e.g., USD).
                - limit: Number of data points (max 2000).
                - aggregate: Aggregation (e.g., 1, 2, 3, etc.).
                
        Returns:
            DataFrame with the fetched data.
        """
        params = params or {}
        
        endpoint = params.get('endpoint', 'histominute')
        fsym = params.get('fsym', 'BTC')
        tsym = params.get('tsym', 'USD')
        limit = params.get('limit', 100)
        aggregate = params.get('aggregate', 1)
        
        # Build URL
        url = f"{self.base_url}{endpoint}"
        
        # Build request parameters
        request_params = {
            'fsym': fsym,
            'tsym': tsym,
            'limit': limit,
            'aggregate': aggregate
        }
        
        # Add API key if available
        headers = {}
        if self.api_key:
            headers['authorization'] = f"Apikey {self.api_key}"
        
        # Make API request
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=request_params, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self.logger.error(f"CryptoCompare API error: {response.status}, {error_text}")
                    raise Exception(f"API request failed with status {response.status}: {error_text}")
                
                data = await response.json()
                
                # Check for API error messages
                if 'Response' in data and data['Response'] == 'Error':
                    self.logger.error(f"CryptoCompare API error: {data.get('Message', 'Unknown error')}")
                    raise Exception(f"API error: {data.get('Message', 'Unknown error')}")
                
                return await self.preprocess_data(data)
    
    async def preprocess_data(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess CryptoCompare data into a pandas DataFrame.
        
        Args:
            data: Raw data from the CryptoCompare API.
            
        Returns:
            Processed DataFrame.
        """
        # Extract data points
        data_points = data.get('Data', [])
        
        if not data_points:
            self.logger.error("No data points found in CryptoCompare response")
            return pd.DataFrame()
            
        # Create DataFrame
        df = pd.DataFrame(data_points)
        
        # Convert time to datetime index
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.set_index('time')
            
        # Sort by timestamp
        df = df.sort_index()
        
        return df
    
    async def get_historical_data(
        self, 
        start_date: Union[str, datetime], 
        end_date: Union[str, datetime] = None,
        params: Dict[str, Any] = None
    ) -> pd.DataFrame:
        """Get historical data from CryptoCompare.
        
        Args:
            start_date: Start date for historical data.
            end_date: End date for historical data (defaults to current time).
            params: Additional parameters for the API request.
            
        Returns:
            DataFrame with historical data.
        """
        params = params or {}
        
        # Convert dates to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
            
        if end_date is None:
            end_date = datetime.now()
            
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        # Calculate time difference and determine appropriate endpoint
        time_diff = (end_date - start_date).days
        
        if time_diff <= 3:
            # For short ranges (up to 3 days), use minute data
            endpoint = 'histominute'
            limit = min(params.get('limit', 2000), 2000)  # Max 2000 data points
            aggregate = params.get('aggregate', 1)
        elif time_diff <= 30:
            # For medium ranges (up to 30 days), use hourly data
            endpoint = 'histohour'
            limit = min(params.get('limit', 2000), 2000)  # Max 2000 data points
            aggregate = params.get('aggregate', 1)
        else:
            # For long ranges, use daily data
            endpoint = 'histoday'
            limit = min(params.get('limit', 2000), 2000)  # Max 2000 data points
            aggregate = params.get('aggregate', 1)
            
        # Update params
        request_params = params.copy()
        request_params['endpoint'] = endpoint
        request_params['limit'] = limit
        request_params['aggregate'] = aggregate
        
        # Fetch data
        data, metadata = await self.get_data(request_params)
        
        # Filter to the requested date range
        if not data.empty:
            data = data[(data.index >= start_date) & (data.index <= end_date)]
            
        return data