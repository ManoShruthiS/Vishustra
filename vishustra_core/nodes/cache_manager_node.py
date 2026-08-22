import logging
from typing import Any, Dict

# Assuming vishustra_core is in PYTHONPATH or installed
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A processing node responsible for managing data caching operations within a Vishustra pipeline.

    This node can perform two primary operations based on the 'cache_operation'
    key provided in the `context`: 'read' or 'write'.

    For 'read' operations:
    - It attempts to retrieve data from the 'cache_store' using the 'cache_key'.
    - If data is found, it returns the cached data and sets 'cache_hit' to True in the context.
    - If not found (cache miss), it returns the original input 'data' to the node
      and sets 'cache_hit' to False, signaling downstream nodes to compute the result.

    For 'write' operations:
    - It stores the input 'data' into the 'cache_store' using the 'cache_key'.
    - It then returns the stored 'data', allowing it to pass to the next node.

    Context Expectations:
    - 'cache_store': A mutable dictionary-like object (e.g., `dict`, `functools.lru_cache` instance,
                     or a custom cache client) where cached data is stored and retrieved. (Required)
    - 'cache_key': The key to use for accessing the cache. This key should uniquely
                   identify the data being cached. (Required)
    - 'cache_operation': A string, either 'read' or 'write', dictating the node's action. (Required)
    - 'cache_hit': A boolean flag, which this node sets during 'read' operations
                   to indicate whether the data was found in the cache.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Manages caching operations (read or write) based on the provided context.

        Args:
            data: The input data for this node.
                  - For 'read' operations: This can be any data that needs to be
                    passed through in case of a cache miss, often the original input
                    to the pipeline or a default value.
                  - For 'write' operations: This is the actual value to be stored
                    in the cache.
            context: A dictionary containing shared state and configuration for the pipeline.
                     Must include 'cache_store', 'cache_key', and 'cache_operation'.

        Returns:
            The result of the cache operation:
            - Cached data if 'read' and a cache hit occurred.
            - The original `data` input if 'read' and a cache miss occurred.
            - The `data` that was written if 'write'.

        Raises:
            ValueError: If 'cache_store', 'cache_key', or 'cache_operation' are
                        missing or invalid in the `context`.
            Exception: Propagates any underlying errors from cache interaction after logging.
        """
        cache_store = context.get('cache_store')
        cache_key = context.get('cache_key')
        cache_operation = context.get('cache_operation')

        # Validate required context keys
        if not (isinstance(cache_store, dict) or
                (hasattr(cache_store, 'get') and hasattr(cache_store, '__setitem__'))):
            raise ValueError(
                "Context missing 'cache_store' or it's not a dictionary-like object "
                "(must support .get() and item assignment)."
            )
        if cache_key is None:
            raise ValueError("Context missing 'cache_key' for cache operation.")
        if cache_operation not in ['read', 'write']:
            raise ValueError(
                f"Context missing 'cache_operation' or it's invalid. "
                f"Expected 'read' or 'write', but got '{cache_operation}'."
            )

        try:
            if cache_operation == 'read':
                return self._read_from_cache(data, cache_store, cache_key, context)
            elif cache_operation == 'write':
                return self._write_to_cache(data, cache_store, cache_key)
        except Exception as e:
            logger.exception(
                f"CacheManagerNode encountered an error during '{cache_operation}' "
                f"operation with key '{cache_key}'."
            )
            # Re-raise the exception to propagate the failure upstream,
            # allowing the orchestration layer to handle it appropriately.
            raise

    def _read_from_cache(self, data: Any, cache_store: Any, cache_key: Any, context: Dict[str, Any]) -> Any:
        """
        Helper method to attempt reading data from the cache.
        """
        cached_value = cache_store.get(cache_key)
        if cached_value is not None:
            context['cache_hit'] = True
            logger.debug(f"Cache hit for key '{cache_key}'.")
            return cached_value
        else:
            context['cache_hit'] = False
            logger.debug(f"Cache miss for key '{cache_key}'. Passing original data through.")
            return data  # In case of a miss, pass the original input data to the next node

    def _write_to_cache(self, data: Any, cache_store: Any, cache_key: Any) -> Any:
        """
        Helper method to write data into the cache.
        """
        cache_store[cache_key] = data
        logger.debug(f"Data successfully written to cache for key '{cache_key}'.")
        return data  # Return the data that was written