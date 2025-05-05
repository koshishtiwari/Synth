import logging
import json
import time
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import requests
from requests.exceptions import RequestException, Timeout
import backoff

from src.config.config_manager import config_manager
from enum import Enum

class DataSourceType(Enum):
    """Enumeration of supported data source types."""
    MARKET = "market"
    ECOMMERCE = "ecommerce"
    IOT = "iot"
    TIMESERIES = "timeseries"
    CUSTOM = "custom"

class DataSourceStatus(Enum):
    """Enumeration of possible data source statuses."""
    READY = "ready"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    INITIALIZING = "initializing"
    UNAVAILABLE = "unavailable"

class BaseDataSource(ABC):
    """Base class for all data sources.
    
    This abstract class defines the interface that all data sources must implement
    to be compatible with the Synth system. It provides methods for fetching data,
    handling rate limits, and caching results.
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """Initialize the data source.
        
        Args:
            name: Name of the data source.
            config: Configuration parameters for the data source.
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.status = DataSourceStatus.INITIALIZING
        self.error_message = None
        
        # Set up rate limiting parameters
        self.rate_limit = self.config.get('rate_limit', {
            'requests_per_minute': 60,
            'burst_limit': 10,
        })
        
        # Set up caching parameters
        self.cache_config = self.config.get('cache', {
            'enabled': True,
            'ttl_seconds': 300,  # 5 minutes
        })
        
        # Set up retry parameters
        self.retry_config = self.config.get('retry', {
            'max_retries': 3,
            'initial_backoff': 1,
            'max_backoff': 60,
        })
        
        # Internal state
        self._last_request_time = 0
        self._request_count = 0
        self._cache = {}
        self._last_successful_response = None
        
        # Rate limiting tracking
        self.rate_limit_config = self.config.get('rate_limit', {})
        self.rate_limit_reset = None
        self.request_count = 0
        self.max_requests = self.rate_limit_config.get('max_requests', 5)
        self.time_window = self.rate_limit_config.get('time_window', 60)  # seconds
        self.last_request_time = None
        
        # Initialize request tracking
        self._reset_request_tracking()
    
    def _reset_request_tracking(self):
        """Reset request tracking for rate limiting."""
        self.request_count = 0
        self.last_request_time = time.time()
    
    def _check_rate_limit(self) -> bool:
        """Check if the data source is currently rate limited.
        
        Returns:
            True if the source is rate limited, False otherwise.
        """
        current_time = time.time()
        
        # If we have a specific reset time, check against that
        if self.rate_limit_reset is not None:
            if current_time < self.rate_limit_reset:
                return True
            else:
                self.rate_limit_reset = None
                self._reset_request_tracking()
                return False
        
        # Otherwise, check based on configured limits
        if self.last_request_time is None:
            self._reset_request_tracking()
            return False
            
        # If we're within the time window, check request count
        elapsed = current_time - self.last_request_time
        if elapsed <= self.time_window:
            if self.request_count >= self.max_requests:
                self.logger.warning(f"Rate limit reached: {self.request_count} requests in {elapsed:.2f} seconds")
                self.status = DataSourceStatus.RATE_LIMITED
                
                # Set the reset time to when the window expires
                self.rate_limit_reset = self.last_request_time + self.time_window
                return True
        else:
            # If we're outside the time window, reset tracking
            self._reset_request_tracking()
            
        return False
    
    def _track_request(self):
        """Track a request for rate limiting purposes."""
        self.request_count += 1
        
        # Update status if we were previously rate limited
        if self.status == DataSourceStatus.RATE_LIMITED:
            self.status = DataSourceStatus.READY
    
    def _update_rate_limit(self, headers: Dict[str, str] = None):
        """Update rate limit information from API response headers.
        
        Args:
            headers: Headers from the API response.
        """
        if not headers:
            return
            
        # Extract rate limit information from headers if available
        # This will be different for each API, so subclasses should override this method
        if 'x-ratelimit-remaining' in headers:
            remaining = int(headers.get('x-ratelimit-remaining', 0))
            
            if remaining <= 0 and 'x-ratelimit-reset' in headers:
                reset_time = int(headers.get('x-ratelimit-reset', 0))
                self.rate_limit_reset = reset_time
                self.status = DataSourceStatus.RATE_LIMITED
                self.logger.warning(f"Rate limit reached, reset at: {datetime.fromtimestamp(reset_time)}")
    
    @abstractmethod
    async def fetch_data(self, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Fetch data from the source.
        
        Args:
            params: Parameters for the data request.
            
        Returns:
            DataFrame with the fetched data.
        """
        pass
    
    async def get_data(self, params: Dict[str, Any] = None, use_cache: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Get data from the source with caching and rate limit handling.
        
        Args:
            params: Parameters for the data request.
            use_cache: Whether to use cached data if available.
            
        Returns:
            Tuple of (DataFrame with data, metadata about the request).
        """
        cache_key = self._make_cache_key(params)
        metadata = {
            'source': self.name,
            'timestamp': datetime.now().isoformat(),
            'params': params,
            'from_cache': False,
            'rate_limited': False,
            'error': None,
        }
        
        # Check cache if enabled
        if use_cache and self.cache_config.get('enabled', True):
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                metadata['from_cache'] = True
                return cached_data, metadata
        
        # Check rate limit
        if not self._can_make_request():
            metadata['rate_limited'] = True
            metadata['error'] = "Rate limit exceeded"
            self.logger.warning(f"Rate limit exceeded for {self.name}, returning last successful response or empty DataFrame")
            
            # Return last successful response if available, otherwise empty DataFrame
            if self._last_successful_response is not None:
                return self._last_successful_response, metadata
            else:
                return pd.DataFrame(), metadata
        
        # Fetch data with retry logic
        try:
            data = await self.fetch_data(params)
            self._update_request_stats()
            
            # Update cache and last successful response
            if self.cache_config.get('enabled', True):
                self._add_to_cache(cache_key, data)
            self._last_successful_response = data
            
            return data, metadata
            
        except Exception as e:
            metadata['error'] = str(e)
            self.logger.error(f"Error fetching data from {self.name}: {e}")
            
            # Return last successful response if available, otherwise empty DataFrame
            if self._last_successful_response is not None:
                return self._last_successful_response, metadata
            else:
                return pd.DataFrame(), metadata
    
    def _make_cache_key(self, params: Dict[str, Any] = None) -> str:
        """Create a cache key from the parameters.
        
        Args:
            params: Parameters for the data request.
            
        Returns:
            String cache key.
        """
        if params is None:
            return "default"
        
        # Convert params to a sorted, stable string representation
        return json.dumps(params, sort_keys=True)
    
    def _get_from_cache(self, key: str) -> Optional[pd.DataFrame]:
        """Get data from the cache.
        
        Args:
            key: Cache key.
            
        Returns:
            DataFrame from cache or None if not found or expired.
        """
        if key not in self._cache:
            return None
        
        cache_entry = self._cache[key]
        cache_time = cache_entry['time']
        ttl = self.cache_config.get('ttl_seconds', 300)
        
        # Check if cache entry is expired
        if (time.time() - cache_time) > ttl:
            del self._cache[key]
            return None
        
        return cache_entry['data']
    
    def _add_to_cache(self, key: str, data: pd.DataFrame) -> None:
        """Add data to the cache.
        
        Args:
            key: Cache key.
            data: Data to cache.
        """
        self._cache[key] = {
            'time': time.time(),
            'data': data.copy()
        }
        
        # Clean up old cache entries
        self._cleanup_cache()
    
    def _cleanup_cache(self) -> None:
        """Remove expired entries from the cache."""
        ttl = self.cache_config.get('ttl_seconds', 300)
        current_time = time.time()
        
        keys_to_delete = []
        for key, entry in self._cache.items():
            if (current_time - entry['time']) > ttl:
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            del self._cache[key]
    
    def _can_make_request(self) -> bool:
        """Check if we can make a request based on rate limiting.
        
        Returns:
            True if a request can be made, False otherwise.
        """
        current_time = time.time()
        requests_per_minute = self.rate_limit.get('requests_per_minute', 60)
        burst_limit = self.rate_limit.get('burst_limit', 10)
        
        # Reset counter if more than a minute has passed
        if current_time - self._last_request_time > 60:
            self._request_count = 0
            self._last_request_time = current_time
            return True
        
        # Check if we're within rate limits
        if self._request_count < min(requests_per_minute, burst_limit):
            return True
        
        return False
    
    def _update_request_stats(self) -> None:
        """Update request statistics after making a request."""
        current_time = time.time()
        
        # Reset counter if more than a minute has passed
        if current_time - self._last_request_time > 60:
            self._request_count = 1
        else:
            self._request_count += 1
            
        self._last_request_time = current_time
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        self._cache = {}
        self.logger.info(f"Cache cleared for {self.name}")
    
    async def get_historical_data(
        self, 
        start_date: Union[str, datetime], 
        end_date: Union[str, datetime] = None,
        params: Dict[str, Any] = None
    ) -> pd.DataFrame:
        """Get historical data from the source.
        
        Args:
            start_date: Start date for historical data.
            end_date: End date for historical data (defaults to current time).
            params: Additional parameters for the data request.
            
        Returns:
            DataFrame with historical data.
        """
        # Implement in subclasses
        raise NotImplementedError("Historical data retrieval not implemented for this data source")

    def get_source_info(self) -> Dict[str, Any]:
        """Get information about the data source.
        
        Returns:
            Dictionary with data source information.
        """
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'rate_limit': self.rate_limit,
            'cache_config': self.cache_config,
        }
    
    @abstractmethod
    async def fetch_historical_data(self, start_date: datetime, end_date: datetime, **params) -> pd.DataFrame:
        """Fetch historical data from the source.
        
        Args:
            start_date: Start date for the data.
            end_date: End date for the data.
            **params: Additional parameters for the request.
            
        Returns:
            DataFrame with the requested data.
        """
        pass
    
    @abstractmethod
    async def fetch_latest_data(self, **params) -> pd.DataFrame:
        """Fetch the latest data point(s) from the source.
        
        Args:
            **params: Parameters for the request.
            
        Returns:
            DataFrame with the latest data.
        """
        pass
    
    async def get_data(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None, **params) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Get data from the source with rate limiting and error handling.
        
        This method checks rate limits before making the request and falls back
        to synthetic data generation if the source is rate limited.
        
        Args:
            start_date: Start date for the data (optional).
            end_date: End date for the data (optional).
            **params: Additional parameters for the request.
            
        Returns:
            Tuple of (DataFrame with the data, metadata dictionary)
        """
        metadata = {
            'source': self.name,
            'timestamp': datetime.now(),
            'data_type': 'real',
            'params': params
        }
        
        # Check if we're rate limited
        if self._check_rate_limit():
            self.logger.info(f"Rate limited, generating synthetic data instead")
            metadata['data_type'] = 'synthetic'
            metadata['reason'] = 'rate_limited'
            
            return await self.generate_synthetic_data(start_date, end_date, **params), metadata
        
        try:
            # Track the request
            self._track_request()
            
            if start_date and end_date:
                # Fetch historical data
                data = await self.fetch_historical_data(start_date, end_date, **params)
                metadata['period'] = f"{start_date} to {end_date}"
            else:
                # Fetch latest data
                data = await self.fetch_latest_data(**params)
                metadata['period'] = 'latest'
            
            self.status = DataSourceStatus.READY
            return data, metadata
            
        except Exception as e:
            self.logger.error(f"Error fetching data: {e}")
            self.error_message = str(e)
            self.status = DataSourceStatus.ERROR
            
            metadata['data_type'] = 'synthetic'
            metadata['reason'] = 'error'
            metadata['error'] = str(e)
            
            return await self.generate_synthetic_data(start_date, end_date, **params), metadata
    
    async def generate_synthetic_data(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None, **params) -> pd.DataFrame:
        """Generate synthetic data when the real source is unavailable.
        
        This method should be implemented by subclasses to provide realistic
        synthetic data that matches the format and characteristics of the real data.
        
        Args:
            start_date: Start date for the data.
            end_date: End date for the data.
            **params: Additional parameters.
            
        Returns:
            DataFrame with synthetic data.
        """
        # Default implementation that returns empty DataFrame
        # Subclasses should override this with realistic data generation
        columns = self.config.get('columns', ['timestamp', 'value'])
        
        if start_date and end_date:
            # Generate time series data
            periods = (end_date - start_date).days + 1
            dates = pd.date_range(start=start_date, periods=periods, freq='D')
            
            # Generate random values
            values = np.random.randn(len(dates))
            
            # Create DataFrame
            data = pd.DataFrame({
                'timestamp': dates,
                'value': values
            })
            
            return data
        else:
            # Generate a single data point
            return pd.DataFrame({
                'timestamp': [datetime.now()],
                'value': [np.random.randn()]
            })
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the data source.
        
        Returns:
            Dictionary with status information.
        """
        status_info = {
            'name': self.name,
            'status': self.status.value,
            'rate_limited': self.status == DataSourceStatus.RATE_LIMITED,
        }
        
        if self.rate_limit_reset:
            status_info['rate_limit_reset'] = datetime.fromtimestamp(self.rate_limit_reset).isoformat()
        
        if self.error_message:
            status_info['error'] = self.error_message
            
        return status_info
    
    def get_source_type(self) -> DataSourceType:
        """Get the type of this data source.
        
        Returns:
            DataSourceType enum value.
        """
        # Default implementation that subclasses should override
        return DataSourceType.CUSTOM