import logging
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class CacheManagerNode(BaseNode):
    """
    A Vishustra processing node designed for managing a data cache.

    This node provides 'get' and 'set' operations for an underlying cache
    store, which must be supplied through the processing context. It acts
    as a robust interface for caching common data or intermediate results
    within an LLM orchestration pipeline, improving performance by reducing
    redundant computations or external API calls.

    Expected `data` structure for 'get' operation:
    ```
    {
        "operation": "get",
        "key": "your_cache_key_identifier"
    }
    ```
    Returns: The cached value if found, or `None` if the key is not present.

    Expected `data` structure for 'set' operation:
    ```
    {
        "operation": "set",
        "key": "your_cache_key_identifier",
        "value": "the_data_to_store_in_cache"
    }
    ```
    Returns: `True` if the value was successfully set, `False` otherwise.

    The actual cache storage mechanism (e.g., an in-memory dictionary,
    a Redis client, an LRU cache, etc.) must be passed into the `context`
    dictionary under the key `"cache_store"`. This object should support
    dictionary-like `get()` and item assignment (`[key] = value`).
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes caching operations (get or set) using the cache store
        provided within the context.

        Args:
            data: A dictionary containing the desired 'operation' (str),
                  'key' (Any), and optionally 'value' (Any) for 'set' operations.
            context: A dictionary expected to contain 'cache_store', which
                     should be a dictionary-like object for caching.

        Returns:
            For 'get' operations: The cached value (Any) or `None` if the
                                   key is not found.
            For 'set' operations: `True` if the value was successfully set,
                                   `False` on failure.

        Raises:
            ValueError: If the `data` or `context` inputs are malformed, or
                        if an unsupported caching operation is requested.
            KeyError: If required keys like 'operation' or 'key' are missing
                      from the `data` payload.
            TypeError: If the 'cache_store' in context is not a dictionary-like object.
        """
        if not isinstance(data, dict):
            logger.error("CacheManagerNode received invalid input data type. Expected a dictionary.")
            raise ValueError("Input 'data' for CacheManagerNode must be a dictionary.")

        operation: Optional[str] = data.get("operation")
        key: Any = data.get("key")

        if operation is None:
            logger.error("Missing 'operation' key in data payload for CacheManagerNode.")
            raise KeyError("The 'operation' key is required in the 'data' dictionary.")
        if key is None:
            logger.error("Missing 'key' key in data payload for CacheManagerNode.")
            raise KeyError("The 'key' key is required in the 'data' dictionary.")

        if "cache_store" not in context:
            logger.error("Context for CacheManagerNode is missing the 'cache_store' object.")
            raise ValueError("CacheManagerNode requires a 'cache_store' object in the 'context'.")

        cache_store: Any = context["cache_store"]
        if not (hasattr(cache_store, 'get') and hasattr(cache_store, '__setitem__')):
            logger.error(
                f"Invalid 'cache_store' type in context. Expected a dictionary-like object "
                f"with 'get' and '__setitem__', got {type(cache_store)}."
            )
            raise TypeError("The 'cache_store' in context must be a dictionary-like object.")

        if operation == "get":
            return self._get_from_cache(key, cache_store)
        elif operation == "set":
            value: Any = data.get("value")
            if value is None:
                logger.error("Missing 'value' key in data payload for 'set' operation.")
                raise KeyError("The 'value' key is required for a 'set' operation.")
            return self._set_to_cache(key, value, cache_store)
        else:
            logger.warning(f"Unsupported cache operation requested: '{operation}'.")
            raise ValueError(f"Unsupported operation: '{operation}'. Expected 'get' or 'set'.")

    def _get_from_cache(self, key: Any, cache_store: Any) -> Any:
        """
        Helper method to retrieve a value from the cache store.

        Args:
            key: The key to look up in the cache.
            cache_store: The cache object.

        Returns:
            The value associated with the key, or None if not found.
        """
        try:
            value = cache_store.get(key)
            if value is not None:
                logger.debug(f"Cache hit for key: '{key}'")
            else:
                logger.debug(f"Cache miss for key: '{key}'")
            return value
        except Exception as e:
            logger.error(f"Failed to retrieve key '{key}' from cache: {e}", exc_info=True)
            # Re-raise to allow upstream handling if retrieval itself failed (e.g., Redis down)
            raise

    def _set_to_cache(self, key: Any, value: Any, cache_store: Any) -> bool:
        """
        Helper method to store a value in the cache store.

        Args:
            key: The key under which to store the value.
            value: The data to store.
            cache_store: The cache object.

        Returns:
            True if the value was successfully stored, False otherwise.
        """
        try:
            cache_store[key] = value
            logger.debug(f"Successfully set key '{key}' in cache.")
            return True
        except Exception as e:
            logger.error(f"Failed to set key '{key}' in cache: {e}", exc_info=True)
            return False