import logging
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A processing node designed to act as an in-memory cache manager.

    This node facilitates common caching operations such as storing, retrieving,
    deleting, and checking for the existence of data items. Operations are
    controlled by specific keys within the `context` dictionary: 'cache_key'
    to identify the data, and 'cache_operation' to specify the action.

    Supported operations via 'cache_operation':
    - "set": Stores the input `data` under the key specified by `context['cache_key']`.
    - "get": Retrieves the value associated with `context['cache_key']`.
             Raises a `KeyError` if the key is not found in the cache.
    - "delete": Removes the entry corresponding to `context['cache_key']`.
                If the key does not exist, a warning is logged, and no action is taken.
    - "has": Checks for the existence of `context['cache_key']` in the cache.
             Returns `True` if the key exists, `False` otherwise.

    Input `context` parameters:
    - 'cache_key' (str): The unique identifier for the cached item. (Required for all operations)
    - 'cache_operation' (str): The desired operation ('set', 'get', 'delete', 'has'). (Required)
    """

    _cache: Dict[str, Any]

    def __init__(self):
        """
        Initializes the CacheManagerNode with an empty in-memory dictionary
        to serve as its cache.
        """
        self._cache = {}
        logger.debug("CacheManagerNode instance initialized with an empty internal cache.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes a cache operation based on the provided `context`.

        Args:
            data (Any): The data to be cached. This argument is primarily
                        relevant for the 'set' operation.
            context (Dict[str, Any]): A dictionary containing configuration
                                      for the cache operation:
                - 'cache_key' (str): The specific key for the cache entry.
                - 'cache_operation' (str): The action to perform ('set', 'get', 'delete', 'has').

        Returns:
            Any: The result of the requested cache operation:
                - For 'get': The retrieved value associated with `cache_key`.
                - For 'set': The `data` that was just stored.
                - For 'delete': `None`.
                - For 'has': A boolean indicating the existence of the `cache_key`.

        Raises:
            ValueError: If 'cache_key' or 'cache_operation' are missing or invalid
                        in the `context`, or if an unsupported 'cache_operation' is provided.
            KeyError: For 'get' operations if the specified 'cache_key' is not found
                      in the cache.
        """
        cache_key = context.get('cache_key')
        cache_operation = context.get('cache_operation')

        if not isinstance(cache_key, str) or not cache_key:
            logger.error("Invalid or missing 'cache_key' in context. Expected a non-empty string.")
            raise ValueError("Context must contain a non-empty string value for 'cache_key'.")

        if not isinstance(cache_operation, str) or not cache_operation:
            logger.error("Invalid or missing 'cache_operation' in context. Expected a non-empty string.")
            raise ValueError("Context must contain a non-empty string value for 'cache_operation'.")

        operation_lower = cache_operation.lower()
        logger.debug(f"Executing cache operation '{operation_lower}' for key '{cache_key}'.")

        if operation_lower == "set":
            self._cache[cache_key] = data
            logger.info(f"Successfully cached data under key: '{cache_key}'.")
            return data
        elif operation_lower == "get":
            if cache_key in self._cache:
                value = self._cache[cache_key]
                logger.debug(f"Retrieved data for key: '{cache_key}'.")
                return value
            else:
                logger.warning(f"Attempted to 'get' non-existent cache key: '{cache_key}'.")
                raise KeyError(f"Cache key '{cache_key}' not found in cache.")
        elif operation_lower == "delete":
            if cache_key in self._cache:
                del self._cache[cache_key]
                logger.info(f"Successfully deleted data for key: '{cache_key}'.")
            else:
                logger.warning(f"Attempted to 'delete' non-existent cache key: '{cache_key}'. No action taken.")
            return None
        elif operation_lower == "has":
            exists = cache_key in self._cache
            logger.debug(f"Existence check for key '{cache_key}': {exists}.")
            return exists
        else:
            logger.error(f"Unsupported cache operation: '{cache_operation}'.")
            raise ValueError(f"Unknown 'cache_operation': '{cache_operation}'. "
                             "Expected one of 'set', 'get', 'delete', or 'has'.")