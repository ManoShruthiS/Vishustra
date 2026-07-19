import logging
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManager(BaseNode):
    """
    A processing node designed for interacting with a cache store within the orchestration context.

    This node provides mechanisms for both reading from and writing to a cache.
    It expects the cache itself to be passed via the `context` dictionary.

    Operations:
    1.  **Read from cache**: If the input `data` dictionary contains only a
        'cache_key' (and no 'value'), the node attempts to retrieve the
        corresponding item from the 'cache_store' located in the `context`.
        -   Returns the cached value if found (cache hit).
        -   Returns `None` if the item is not found (cache miss), allowing
            downstream nodes to compute the value.
    2.  **Write to cache**: If the input `data` dictionary contains both a
        'cache_key' and a 'value', the node stores the 'value' under the
        specified 'cache_key' in the 'cache_store' in the `context`.
        -   Returns the value that was successfully stored.

    The 'cache_store' is expected to be a dictionary-like object (e.g., a `dict`
    or a custom cache implementation) available under the key 'cache_store'
    within the `context` dictionary.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to perform either a cache read or a cache write operation.

        Args:
            data: A dictionary containing the operation details.
                  - Must contain a 'cache_key' (str).
                  - If it also contains a 'value' (Any), it's considered a write operation.
                  - Otherwise, it's a read operation.
            context: A dictionary of shared resources, expected to contain a
                     'cache_store' key which holds the cache dictionary.

        Returns:
            - The retrieved value on a successful cache read (hit).
            - `None` on a cache read miss.
            - The value that was written on a successful cache write.

        Raises:
            ValueError: If the input `data` is malformed (e.g., not a dict,
                        missing 'cache_key', or 'cache_key' is not a string).
            RuntimeError: If the 'cache_store' is not found or is invalid within
                          the `context`, or if an underlying error occurs during
                          interaction with the cache.
        """
        if not isinstance(data, dict):
            logger.error("CacheManager received invalid input: data must be a dictionary.")
            raise ValueError("Input 'data' for CacheManager must be a dictionary.")

        cache_key = data.get("cache_key")
        if not isinstance(cache_key, str) or not cache_key:
            logger.error(f"CacheManager requires a non-empty string 'cache_key' in input data. Received: {data}")
            raise ValueError("Missing or invalid 'cache_key' in input data for CacheManager.")

        cache_store = context.get("cache_store")
        if not isinstance(cache_store, dict): # We expect a dict-like interface for the cache.
            logger.error(
                "Cache store 'cache_store' not found or is invalid in context. "
                "Expected a dictionary-like object for cache operations."
            )
            raise RuntimeError("Cache store 'cache_store' not properly initialized in context.")

        if "value" in data:
            # This signifies a write operation: store the provided value.
            value_to_cache = data["value"]
            try:
                cache_store[cache_key] = value_to_cache
                logger.info(f"CacheManager: Stored item for key '{cache_key}'.")
                return value_to_cache
            except Exception as e:
                logger.exception(f"CacheManager: Failed to store item with key '{cache_key}' in cache.")
                raise RuntimeError(f"Failed to write to cache for key '{cache_key}': {e}")
        else:
            # This signifies a read operation: attempt to retrieve the value.
            try:
                cached_value = cache_store.get(cache_key)
                if cached_value is not None:
                    logger.info(f"CacheManager: Cache hit for key '{cache_key}'.")
                    return cached_value
                else:
                    logger.info(f"CacheManager: Cache miss for key '{cache_key}'.")
                    return None  # Explicitly return None to signal a cache miss
            except Exception as e:
                logger.exception(f"CacheManager: Failed to retrieve item with key '{cache_key}' from cache.")
                raise RuntimeError(f"Failed to read from cache for key '{cache_key}': {e}")