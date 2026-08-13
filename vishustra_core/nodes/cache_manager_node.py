import logging
from typing import Any, Dict, Optional

# Assuming this path exists in the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A Vishustra node that manages an in-memory cache.
    It supports 'get' and 'set' operations for data caching.

    The 'process' method expects 'data' to serve as the key for 'get' operations,
    and as the value to cache for 'set' operations.
    The 'context' dictionary must specify the 'operation' (either "get" or "set")
    and, for "set" operations, explicitly provide the 'key'.
    """

    def __init__(self, initial_cache: Optional[Dict[Any, Any]] = None):
        """
        Initializes the CacheManagerNode with an optional initial cache state.

        Args:
            initial_cache: An optional dictionary to pre-populate the cache.
                           If None, an empty dictionary is used.
        """
        self._cache: Dict[Any, Any] = initial_cache if initial_cache is not None else {}
        logger.debug(f"CacheManagerNode initialized with {len(self._cache)} initial items.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes a cache operation (get or set) based on the provided 'context'.

        For a 'get' operation:
            - `data` should be the cache key to retrieve.
            - `context` must contain `{"operation": "get"}`.
            - Returns the cached value if found, otherwise `None`.

        For a 'set' operation:
            - `data` should be the value to store in the cache.
            - `context` must contain `{"operation": "set", "key": <cache_key>}`.
            - Returns the value that was successfully set.

        Args:
            data: The cache key for a 'get' operation, or the value to cache for a 'set' operation.
            context: A dictionary containing the operation type and additional parameters.

        Returns:
            The retrieved value for 'get', the set value for 'set', or `None` for a cache miss on 'get'.

        Raises:
            ValueError: If the 'operation' in 'context' is invalid, or if required
                        parameters (like 'key' for a 'set' operation) are missing.
        """
        operation = context.get("operation")

        if operation == "get":
            key_to_get = data  # For 'get', 'data' is interpreted as the key.
            logger.debug(f"Attempting to retrieve key '{key_to_get}' from cache.")
            cached_value = self._cache.get(key_to_get)
            if cached_value is not None:
                logger.info(f"Cache HIT for key '{key_to_get}'.")
                return cached_value
            else:
                logger.info(f"Cache MISS for key '{key_to_get}'.")
                return None
        elif operation == "set":
            key_to_set = context.get("key")
            value_to_set = data  # For 'set', 'data' is interpreted as the value.

            if key_to_set is None:
                logger.error("Attempted 'set' operation without a 'key' specified in context.")
                raise ValueError("CacheManagerNode 'set' operation requires a 'key' in context.")
            
            self._cache[key_to_set] = value_to_set
            logger.info(f"Cache SET operation successful for key '{key_to_set}'. Value type: {type(value_to_set).__name__}.")
            return value_to_set
        else:
            logger.error(
                f"Invalid or missing 'operation' in context: '{operation}'. "
                f"Context provided: {context}"
            )
            raise ValueError(
                "CacheManagerNode requires 'operation' in context to be either 'get' or 'set'."
            )
