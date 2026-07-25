import logging
import time
from typing import Any, Dict, Optional, Tuple

# Importing BaseNode from the specified project path
try:
    from vishustra_core.nodes.base_node import BaseNode
except ImportError:
    # Fallback for local development environments where the full package
    # structure might not be available yet.
    from abc import ABC, abstractmethod

    class BaseNode(ABC):
        @abstractmethod
        def process(self, data: Any, context: Dict[str, Any]) -> Any:
            """Processes the input data and returns the result."""
            pass
            
        @property
        @abstractmethod
        def node_name(self) -> str:
            """Returns the name of the node."""
            pass

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A processing node designed to manage an in-memory cache.
    It supports 'get', 'set', and 'invalidate' operations on cache entries,
    including an optional Time-To-Live (TTL) for 'set' operations.

    The node's behavior is dictated by the 'cache_action' key within the context dictionary:
    - 'get': Attempts to retrieve data from the cache using 'cache_key'.
             Returns the cached data if found and not expired, otherwise None.
    - 'set': Stores the input 'data' under 'cache_key' in the cache.
             An optional 'ttl_seconds' can be provided in context for expiration.
             Returns the data that was successfully stored.
    - 'invalidate': Removes an entry identified by 'cache_key' from the cache.
                    Returns True if the entry was removed, False if it didn't exist.
    """

    # Internal in-memory cache storage: {cache_key: (value, expiry_timestamp_float_or_None)}
    _cache: Dict[str, Tuple[Any, Optional[float]]]

    def __init__(self):
        """Initializes the CacheManagerNode and its internal cache."""
        super().__init__()
        self._cache = {}
        logger.info("CacheManagerNode initialized with an in-memory cache.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "CacheManager"

    def _is_expired(self, expiry_timestamp: Optional[float]) -> bool:
        """
        Checks if a cache entry has expired based on its timestamp.

        Args:
            expiry_timestamp: The absolute timestamp when the entry expires, or None for no expiration.

        Returns:
            True if the entry has expired, False otherwise.
        """
        if expiry_timestamp is None:
            return False
        return time.time() > expiry_timestamp

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes a cache management operation based on the provided context.

        Args:
            data: The input data. For 'set' actions, this is the value to cache.
                  For 'get' and 'invalidate', this parameter is typically ignored.
            context: A dictionary containing operational parameters:
                     - 'cache_key' (str): The unique key for the cache operation. (Required)
                     - 'cache_action' (str): The specific action to perform ('get', 'set', 'invalidate'). (Required)
                     - 'ttl_seconds' (float, optional): Time-To-Live in seconds for 'set' actions.

        Returns:
            Any: The result of the cache operation, which varies by action:
                 - 'get': The cached data if available and valid, otherwise None.
                 - 'set': The 'data' that was successfully stored in the cache.
                 - 'invalidate': True if the entry was successfully removed, False otherwise.

        Raises:
            ValueError: If 'cache_key' or 'cache_action' are missing, invalid, or unsupported.
            TypeError: If 'ttl_seconds' is provided but is not a valid numeric type.
        """
        cache_key = context.get("cache_key")
        cache_action = context.get("cache_action")

        if not isinstance(cache_key, str) or not cache_key.strip():
            logger.error("Context error: 'cache_key' must be a non-empty string for CacheManagerNode.")
            raise ValueError("CacheManagerNode requires a valid 'cache_key' (non-empty string) in context.")

        if not isinstance(cache_action, str) or not cache_action.strip():
            logger.error(f"Context error for key '{cache_key}': 'cache_action' must be a non-empty string.")
            raise ValueError("CacheManagerNode requires a valid 'cache_action' (non-empty string) in context.")

        action = cache_action.lower()

        if action == "get":
            return self._handle_get(cache_key)
        elif action == "set":
            ttl_seconds = context.get("ttl_seconds")
            return self._handle_set(cache_key, data, ttl_seconds)
        elif action == "invalidate":
            return self._handle_invalidate(cache_key)
        else:
            logger.error(f"Unsupported cache action '{cache_action}' for key '{cache_key}'.")
            raise ValueError(f"Unsupported cache action: '{cache_action}'. Must be 'get', 'set', or 'invalidate'.")

    def _handle_get(self, cache_key: str) -> Any:
        """
        Handles the 'get' cache action, retrieving data and checking for expiration.

        Args:
            cache_key: The key of the entry to retrieve.

        Returns:
            The cached value if found and not expired, otherwise None.
        """
        entry = self._cache.get(cache_key)
        if entry is None:
            logger.debug(f"Cache miss for key: '{cache_key}'.")
            return None

        value, expiry_timestamp = entry
        if self._is_expired(expiry_timestamp):
            del self._cache[cache_key]  # Clean up expired entry
            logger.debug(f"Cache entry for key '{cache_key}' expired and was removed (miss).")
            return None
        
        logger.debug(f"Cache hit for key: '{cache_key}'.")
        return value

    def _handle_set(self, cache_key: str, value: Any, ttl_seconds: Optional[float]) -> Any:
        """
        Handles the 'set' cache action, storing data with an optional TTL.

        Args:
            cache_key: The key under which to store the value.
            value: The data to be cached.
            ttl_seconds: Optional Time-To-Live in seconds for the entry.

        Returns:
            The value that was stored.

        Raises:
            TypeError: If 'ttl_seconds' is provided but is not a non-negative number.
        """
        expiry_timestamp: Optional[float] = None
        if ttl_seconds is not None:
            if not isinstance(ttl_seconds, (int, float)) or ttl_seconds < 0:
                logger.error(
                    f"Invalid 'ttl_seconds' value '{ttl_seconds}' for key '{cache_key}'. "
                    "Must be a non-negative number."
                )
                raise TypeError("Invalid 'ttl_seconds'. Must be a non-negative number.")
            expiry_timestamp = time.time() + ttl_seconds
        
        self._cache[cache_key] = (value, expiry_timestamp)
        logger.debug(f"Cache entry set for key: '{cache_key}' with TTL: {ttl_seconds} seconds.")
        return value

    def _handle_invalidate(self, cache_key: str) -> bool:
        """
        Handles the 'invalidate' cache action, removing an entry.

        Args:
            cache_key: The key of the entry to invalidate.

        Returns:
            True if the entry was removed, False if it did not exist.
        """
        if cache_key in self._cache:
            del self._cache[cache_key]
            logger.debug(f"Cache entry invalidated for key: '{cache_key}'.")
            return True
        logger.debug(f"Attempted to invalidate non-existent cache key: '{cache_key}'.")
        return False