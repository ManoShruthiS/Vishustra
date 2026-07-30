import time
import logging
from typing import Any, Dict, Optional, Tuple

# Assuming BaseNode is located here as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A Vishustra node designed to manage data caching operations (get, set, delete)
    within the orchestration flow.

    It expects the 'data' input to be a dictionary specifying the desired operation,
    the cache key, and optionally a value and time-to-live (TTL).

    The actual cache store (e.g., an in-memory dictionary or a client to a
    distributed cache) is expected to be provided in the 'context' dictionary
    under the key 'cache_store'. This allows for flexible cache implementation
    and sharing across nodes.

    Data Input Format:
    A dictionary with the following keys:
    - "operation": str, required. Must be "get", "set", or "delete".
    - "key": Any, required. The identifier for the cache entry.
    - "value": Any, required for "set" operation. The data to be cached.
    - "ttl": Optional[int], optional for "set" operation. Time-to-live in seconds.
             If provided and positive, the entry will expire after this duration.
             If None, the entry does not expire.

    Context Expected:
    A dictionary containing:
    - "cache_store": Dict[Any, Tuple[Any, Optional[float]]], required.
                     A mutable dictionary where keys map to (value, expiry_timestamp) tuples.
                     `expiry_timestamp` is a Unix timestamp (float) or None for no expiry.

    Example Usage:
    # To set a value
    data = {"operation": "set", "key": "user_data", "value": {"name": "Alice"}, "ttl": 300}
    # To get a value
    data = {"operation": "get", "key": "user_data"}
    # To delete a value
    data = {"operation": "delete", "key": "user_data"}
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "CacheManagerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes cache operations (get, set, delete) based on the provided data.

        Args:
            data: A dictionary specifying the cache operation, key, value (for set),
                  and optional TTL (for set).
            context: A dictionary containing shared resources, crucially 'cache_store'.

        Returns:
            The result of the operation:
            - For "get": The cached value if found and not expired; otherwise, None.
            - For "set": True upon successful caching.
            - For "delete": True if the key was found and deleted; otherwise, False.

        Raises:
            ValueError: If the input 'data' is malformed, missing required fields,
                        or contains an unknown operation type.
            RuntimeError: If 'cache_store' is not properly configured or found in context.
        """
        if not isinstance(data, dict):
            logger.error("CacheManagerNode received invalid data input: Expected a dictionary.")
            raise ValueError("Invalid data input: Expected a dictionary for cache operation details.")

        operation = data.get("operation")
        key = data.get("key")

        if operation not in ["get", "set", "delete"]:
            logger.error(f"CacheManagerNode received an unknown operation: '{operation}'.")
            raise ValueError(f"Unknown cache operation: '{operation}'. Expected 'get', 'set', or 'delete'.")
        
        if key is None:
            logger.error(f"CacheManagerNode operation '{operation}' requires a 'key'.")
            raise ValueError(f"Missing 'key' for '{operation}' operation.")

        cache_store = context.get("cache_store")
        if not isinstance(cache_store, dict):
            logger.error("CacheManagerNode requires a mutable dictionary 'cache_store' in context.")
            raise RuntimeError("CacheManagerNode: 'cache_store' not found or invalid in context. "
                               "Expected a dictionary for cache storage.")

        current_time = time.time()

        if operation == "get":
            return self._handle_get(key, cache_store, current_time)
        elif operation == "set":
            value = data.get("value")
            ttl = data.get("ttl")
            return self._handle_set(key, value, ttl, cache_store, current_time)
        elif operation == "delete":
            return self._handle_delete(key, cache_store)
        
        # This part should ideally be unreachable due to the initial operation check,
        # but included for defensive programming.
        logger.critical(f"CacheManagerNode reached an unexpected state with operation: {operation}")
        raise RuntimeError(f"Unexpected state in CacheManagerNode for operation: {operation}")


    def _handle_get(self, key: Any, cache_store: Dict[Any, Tuple[Any, Optional[float]]], current_time: float) -> Any:
        """Helper to handle 'get' operation."""
        logger.debug(f"CacheManagerNode: Attempting to get key '{key}'.")
        cached_item = cache_store.get(key)

        if cached_item is None:
            logger.debug(f"Cache miss for key '{key}'.")
            return None
        
        cached_value, expiry_time = cached_item
        if expiry_time is not None and current_time > expiry_time:
            logger.info(f"Cache entry for key '{key}' expired. Removing from cache.")
            del cache_store[key]
            return None
        
        logger.debug(f"Cache hit for key '{key}'.")
        return cached_value

    def _handle_set(self, key: Any, value: Any, ttl: Optional[int],
                     cache_store: Dict[Any, Tuple[Any, Optional[float]]], current_time: float) -> bool:
        """Helper to handle 'set' operation."""
        if value is None:
            logger.error("CacheManagerNode 'set' operation requires a 'value'.")
            raise ValueError("Missing 'value' for 'set' operation.")
        
        expiry_time: Optional[float] = None
        if ttl is not None:
            if not isinstance(ttl, (int, float)) or ttl <= 0:
                logger.error(f"Invalid TTL value '{ttl}' for key '{key}'. Must be a positive number.")
                raise ValueError(f"Invalid 'ttl' for 'set' operation. Must be a positive number.")
            expiry_time = current_time + ttl

        cache_store[key] = (value, expiry_time)
        logger.info(f"CacheManagerNode: Key '{key}' set with TTL {ttl} (expires at {expiry_time if expiry_time else 'never'}).")
        return True

    def _handle_delete(self, key: Any, cache_store: Dict[Any, Any]) -> bool:
        """Helper to handle 'delete' operation."""
        if key in cache_store:
            del cache_store[key]
            logger.info(f"CacheManagerNode: Key '{key}' deleted from cache.")
            return True
        logger.debug(f"CacheManagerNode: Key '{key}' not found in cache for deletion.")
        return False
