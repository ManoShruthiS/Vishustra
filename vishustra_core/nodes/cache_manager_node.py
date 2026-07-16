import logging
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

# Assuming vishustra_core.nodes.base_node exists and contains BaseNode
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A Vishustra node responsible for managing an in-memory key-value cache.

    This node supports operations like 'get', 'set', 'invalidate', and 'clear'
    with optional Time-To-Live (TTL) for cached items.

    Operations are controlled via the 'cache_op' key in the context dictionary.

    Context keys for operations:
    - 'cache_op' (str): The desired cache operation ('get', 'set', 'invalidate', 'clear').
      This key is mandatory.

    Data format for operations:
    - 'get': `data` should be the key (Any) to retrieve.
    - 'set': `data` should be a dictionary `{'key': Any, 'value': Any, 'ttl': Optional[float]}`.
             'ttl' is in seconds. If not provided or None, uses `default_ttl` if set,
             otherwise item will not expire.
    - 'invalidate': `data` should be the key (Any) to remove from the cache.
    - 'clear': `data` is ignored.

    Returns:
    - 'get': The cached value if found and not expired, otherwise None.
    - 'set': The value that was stored.
    - 'invalidate': True if the item was found and removed, False otherwise.
    - 'clear': True.
    """

    _cache: Dict[Any, Dict[str, Any]]
    _cache_lock: threading.Lock
    _default_ttl_seconds: Optional[float]

    def __init__(self, default_ttl_seconds: Optional[float] = None):
        """
        Initializes the CacheManagerNode with an optional default Time-To-Live.

        Args:
            default_ttl_seconds (Optional[float]): The default TTL for items
                                                    in seconds. If None, items
                                                    set without a specific TTL
                                                    will not expire.
        """
        self._cache = {}
        self._cache_lock = threading.Lock()
        if default_ttl_seconds is not None and not isinstance(default_ttl_seconds, (int, float)) or default_ttl_seconds < 0:
            logger.warning(
                f"Invalid default_ttl_seconds '{default_ttl_seconds}' provided. "
                "Default TTL will be ignored and set to None."
            )
            self._default_ttl_seconds = None
        else:
            self._default_ttl_seconds = default_ttl_seconds
        logger.debug(f"CacheManagerNode initialized with default_ttl_seconds: {self._default_ttl_seconds}")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "CacheManager"

    def _is_expired(self, cached_item: Dict[str, Any]) -> bool:
        """Checks if a cached item has expired."""
        if 'timestamp' not in cached_item or 'ttl' not in cached_item:
            logger.warning("Cached item missing 'timestamp' or 'ttl' metadata. Treating as not expired.")
            return False # Should not happen with proper 'set' operation
        
        ttl = cached_item['ttl']
        if ttl is None:
            return False  # No TTL, never expires

        stored_time: datetime = cached_item['timestamp']
        expiry_time = stored_time + timedelta(seconds=ttl)
        return datetime.now() > expiry_time

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes cache operations based on the 'cache_op' in the context.

        Args:
            data (Any): Input data, varies by operation.
            context (Dict[str, Any]): Context dictionary containing 'cache_op'.

        Returns:
            Any: The result of the cache operation (e.g., cached value, True/False).

        Raises:
            ValueError: If 'cache_op' is missing or invalid, or if required
                        data fields for an operation are missing/invalid.
        """
        cache_op = context.get('cache_op')
        if not cache_op:
            logger.error("Context is missing the required 'cache_op' key for CacheManagerNode.")
            raise ValueError("Missing 'cache_op' in context for CacheManagerNode.")

        with self._cache_lock:
            if cache_op == 'get':
                key = data
                cached_item = self._cache.get(key)
                if cached_item is None:
                    logger.debug(f"Cache miss for key: '{key}'")
                    return None
                
                if self._is_expired(cached_item):
                    logger.debug(f"Cache hit for key: '{key}' but item expired. Removing.")
                    del self._cache[key]
                    return None
                
                logger.debug(f"Cache hit for key: '{key}'")
                return cached_item['value']

            elif cache_op == 'set':
                if not isinstance(data, dict) or 'key' not in data or 'value' not in data:
                    logger.error(
                        f"Invalid data format for 'set' operation. "
                        f"Expected {{'key': ..., 'value': ..., 'ttl': Optional[float]}}, got: {data}"
                    )
                    raise ValueError(f"Invalid data for 'set' operation in CacheManagerNode: {data}")

                key = data['key']
                value = data['value']
                
                # Use data's ttl if present, else default_ttl_seconds
                item_ttl = data.get('ttl', self._default_ttl_seconds)
                if item_ttl is not None and (not isinstance(item_ttl, (int, float)) or item_ttl < 0):
                     logger.warning(
                        f"Invalid TTL '{item_ttl}' for key '{key}'. "
                        "Setting item without TTL. Ensure TTL is a non-negative number."
                    )
                     item_ttl = None # Treat as no TTL
                
                self._cache[key] = {
                    'value': value,
                    'timestamp': datetime.now(),
                    'ttl': item_ttl
                }
                logger.info(f"Key '{key}' set in cache with TTL: {item_ttl}s")
                return value

            elif cache_op == 'invalidate':
                key = data
                if key in self._cache:
                    del self._cache[key]
                    logger.info(f"Key '{key}' invalidated from cache.")
                    return True
                else:
                    logger.debug(f"Attempted to invalidate non-existent key: '{key}'")
                    return False

            elif cache_op == 'clear':
                self._cache.clear()
                logger.info("Cache entirely cleared.")
                return True

            else:
                logger.error(f"Unknown cache operation '{cache_op}' received by CacheManagerNode.")
                raise ValueError(f"Unknown 'cache_op': '{cache_op}'")

# Example of how to set up logging if this were the main script
# (not part of the node's responsibility but good practice)
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     
#     cache_manager = CacheManagerNode(default_ttl_seconds=5)
#     
#     # Test Set Operation
#     print("\n--- Testing Set ---")
#     data_to_set = {'key': 'my_data', 'value': {'foo': 'bar'}}
#     context_set = {'cache_op': 'set'}
#     result_set = cache_manager.process(data_to_set, context_set)
#     print(f"Set result: {result_set}")
#     
#     # Test Get Operation (hit)
#     print("\n--- Testing Get (Hit) ---")
#     data_to_get = 'my_data'
#     context_get = {'cache_op': 'get'}
#     result_get = cache_manager.process(data_to_get, context_get)
#     print(f"Get result (hit): {result_get}")
#     
#     # Test Get Operation (miss)
#     print("\n--- Testing Get (Miss) ---")
#     data_to_get_miss = 'non_existent_data'
#     result_get_miss = cache_manager.process(data_to_get_miss, context_get)
#     print(f"Get result (miss): {result_get_miss}")
#     
#     # Test Set with explicit TTL
#     print("\n--- Testing Set with explicit TTL ---")
#     data_with_ttl = {'key': 'temp_data', 'value': 'this expires soon', 'ttl': 2}
#     cache_manager.process(data_with_ttl, context_set)
#     
#     # Test Get before TTL expires
#     print("\n--- Testing Get before TTL expires ---")
#     print(f"Get 'temp_data': {cache_manager.process('temp_data', context_get)}")
#     
#     # Wait for TTL to expire
#     import time
#     time.sleep(2.5)
#     
#     # Test Get after TTL expires
#     print("\n--- Testing Get after TTL expires ---")
#     print(f"Get 'temp_data': {cache_manager.process('temp_data', context_get)}")
#     
#     # Test Invalidate
#     print("\n--- Testing Invalidate ---")
#     data_to_invalidate = 'my_data'
#     context_invalidate = {'cache_op': 'invalidate'}
#     result_invalidate = cache_manager.process(data_to_invalidate, context_invalidate)
#     print(f"Invalidate result: {result_invalidate}")
#     print(f"Get 'my_data' after invalidate: {cache_manager.process(data_to_get, context_get)}")
#     
#     # Test Clear
#     print("\n--- Testing Clear ---")
#     data_to_clear = None # data is ignored for clear
#     context_clear = {'cache_op': 'clear'}
#     result_clear = cache_manager.process(data_to_clear, context_clear)
#     print(f"Clear result: {result_clear}")
#     print(f"Get 'non_existent_data' after clear: {cache_manager.process(data_to_get_miss, context_get)}")
#     
#     # Test Error Handling (missing 'cache_op')
#     print("\n--- Testing Error Handling (missing 'cache_op') ---")
#     try:
#         cache_manager.process("some_key", {})
#     except ValueError as e:
#         print(f"Caught expected error: {e}")
#         
#     # Test Error Handling (invalid 'cache_op')
#     print("\n--- Testing Error Handling (invalid 'cache_op') ---")
#     try:
#         cache_manager.process("some_key", {'cache_op': 'unknown'})
#     except ValueError as e:
#         print(f"Caught expected error: {e}")
#         
#     # Test Error Handling (invalid 'set' data)
#     print("\n--- Testing Error Handling (invalid 'set' data) ---")
#     try:
#         cache_manager.process({'key': 'bad'}, {'cache_op': 'set'}) # Missing value
#     except ValueError as e:
#         print(f"Caught expected error: {e}")

