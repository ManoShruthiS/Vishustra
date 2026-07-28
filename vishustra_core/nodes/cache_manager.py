import logging
import time
from typing import Any, Dict, Optional

# Assuming vishustra_core.nodes.base_node is the correct path for BaseNode
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A Vishustra processing node that manages an in-memory cache,
    providing capabilities for storing, retrieving, and deleting data
    with optional time-to-live (TTL) functionality.

    This node enhances data processing pipelines by reducing redundant
    computations or external API calls for frequently accessed data.
    """

    def __init__(self, default_ttl_seconds: Optional[int] = 300):
        """
        Initializes the CacheManagerNode.

        Args:
            default_ttl_seconds: The default time-to-live for cache items in seconds.
                                 If None, items stored without an explicit TTL will
                                 not expire. Defaults to 300 seconds (5 minutes).
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._default_ttl_seconds = default_ttl_seconds
        logger.debug(f"CacheManagerNode initialized with default_ttl_seconds: {default_ttl_seconds}")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "CacheManager"

    def _is_expired(self, key: str) -> bool:
        """
        Checks if a cached item associated with the given key has expired.
        This method assumes the key exists in the cache.
        """
        item = self._cache.get(key)
        if not item:
            # This case indicates an internal inconsistency if called for an existing key
            # but is handled gracefully by _get_item checking for key existence first.
            logger.warning(f"Internal error: _is_expired called for non-existent key '{key}'.")
            return True # Treat as expired if not found

        expiry_time = item.get('expiry_time')
        if expiry_time is None:
            return False  # No expiry time means the item never expires

        return time.time() >= expiry_time

    def _get_item(self, key: str) -> Optional[Any]:
        """
        Retrieves an item from the cache if it exists and is not expired.
        If expired, the item is removed from the cache.
        """
        if key not in self._cache:
            logger.debug(f"Cache miss for key: '{key}' (item not found).")
            return None

        if self._is_expired(key):
            logger.debug(f"Cache item for key '{key}' found but is expired. Removing and returning None.")
            del self._cache[key]
            return None

        logger.debug(f"Cache hit for key: '{key}'.")
        return self._cache[key]['value']

    def _set_item(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> Any:
        """
        Stores an item in the cache with an optional time-to-live.
        If `ttl_seconds` is not provided, the node's `default_ttl_seconds` is used.
        """
        effective_ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl_seconds
        expiry_time = time.time() + effective_ttl if effective_ttl is not None else None

        self._cache[key] = {'value': value, 'expiry_time': expiry_time}
        logger.debug(f"Cache set for key '{key}'. TTL: {effective_ttl}s (expires at {expiry_time}).")
        return value

    def _delete_item(self, key: str) -> bool:
        """
        Deletes an item from the cache if it exists.
        Returns True if deleted, False otherwise.
        """
        if key in self._cache:
            del self._cache[key]
            logger.debug(f"Cache item for key '{key}' successfully deleted.")
            return True
        logger.debug(f"Attempted to delete non-existent cache item for key '{key}'. No action taken.")
        return False

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes a cache operation ('get', 'set', or 'delete') based on the input data.

        The `data` input must be a dictionary with the following structure:
        - 'operation' (str): The desired cache action ('get', 'set', 'delete').
        - 'key' (str): The unique identifier for the cache item.
        - 'value' (Any, optional): Required for 'set' operations, the data to store.
        - 'ttl' (int, optional): For 'set' operations, an item-specific TTL in seconds,
                                 overriding the node's `default_ttl_seconds`.

        Args:
            data: A dictionary specifying the cache operation and its parameters.
            context: A dictionary containing shared pipeline context (not directly used by this node).

        Returns:
            - For 'get': The cached value if found and not expired, otherwise None.
            - For 'set': The value that was just cached.
            - For 'delete': True if the item was deleted, False otherwise.

        Raises:
            TypeError: If the input `data` is not a dictionary.
            ValueError: If 'operation' or 'key' are missing/invalid, or 'value' is missing
                        for a 'set' operation, or 'ttl' is not an integer.
        """
        if not isinstance(data, dict):
            logger.error("Input data for CacheManagerNode must be a dictionary.")
            raise TypeError("Input data for CacheManagerNode must be a dictionary.")

        operation = data.get('operation')
        key = data.get('key')

        if not isinstance(operation, str) or operation.strip() == "":
            logger.error("Missing or invalid 'operation' in input data. Expected 'get', 'set', or 'delete'.")
            raise ValueError("Missing or invalid 'operation'. Must be 'get', 'set', or 'delete'.")
        
        if not isinstance(key, str) or key.strip() == "":
            logger.error("Missing or invalid 'key' in input data.")
            raise ValueError("Missing or invalid 'key'.")

        operation = operation.lower().strip()

        if operation == 'get':
            return self._get_item(key)
        
        elif operation == 'set':
            # Check if 'value' key is present, allowing None as a valid value
            if 'value' not in data:
                logger.error(f"Missing 'value' for 'set' operation with key: '{key}'.")
                raise ValueError("Missing 'value' for 'set' operation.")
            value = data['value']
            
            ttl = data.get('ttl')
            if ttl is not None and not isinstance(ttl, int):
                logger.error(f"Invalid 'ttl' type for 'set' operation with key: '{key}'. Expected int, got {type(ttl).__name__}.")
                raise ValueError("Invalid 'ttl' type. Must be an integer.")
            
            return self._set_item(key, value, ttl)
        
        elif operation == 'delete':
            return self._delete_item(key)
        
        else:
            logger.error(f"Unknown cache operation '{operation}' for key: '{key}'.")
            raise ValueError(f"Unknown operation: '{operation}'. Expected 'get', 'set', or 'delete'.")