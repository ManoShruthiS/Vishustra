import logging
from typing import Any, Dict, Optional

# Assuming this path based on the project context's BaseNode import
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A processing node designed to manage an in-memory cache for data.

    This node supports 'set', 'get', and 'clear' operations on its internal
    cache, facilitating efficient reuse of processed or retrieved data within
    an orchestration flow. Operations are dictated by the 'operation' key
    in the context dictionary.
    """

    def __init__(self):
        super().__init__()
        self._cache: Dict[str, Any] = {}
        logger.debug("CacheManagerNode initialized with an empty in-memory cache.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes a cache operation based on the provided context.

        The `context` dictionary must contain an 'operation' key, specifying
        whether to 'get', 'set', or 'clear' cache entries. For 'get' and 'set'
        operations, a 'key' is also required.

        Context keys:
        - 'operation': (str) Required. Must be one of 'get', 'set', or 'clear'.
        - 'key': (str) Required for 'get' and 'set'. Optional for 'clear' (clears all if not provided).
        - 'default_value': (Any) Optional. The value to return if a 'get' operation
                           finds no entry for the specified key.

        Args:
            data: The data to be cached (relevant for 'set' operation) or ignored
                  (for 'get'/'clear' operations).
            context: A dictionary containing operational parameters for the cache.

        Returns:
            - For 'set': The data that was successfully stored in the cache.
            - For 'get': The cached data, or `default_value` if specified and not found,
                         or `None` if not found and no `default_value` was provided.
            - For 'clear': `True` if the cache (or a specific key) was successfully cleared,
                           `False` otherwise (e.g., attempting to clear a non-existent key).

        Raises:
            ValueError: If 'operation' is missing or invalid, or if 'key' is
                        missing for 'get'/'set' operations or not a string.
        """
        operation: Optional[str] = context.get('operation')
        key: Optional[str] = context.get('key')
        default_value: Any = context.get('default_value')  # For 'get' operations

        if not operation:
            logger.error("CacheManagerNode: 'operation' key is missing in context.")
            raise ValueError("Cache operation 'operation' must be specified in context.")

        operation = operation.lower()

        if operation == 'set':
            if not isinstance(key, str):
                logger.error("CacheManagerNode: 'key' must be a string for 'set' operation.")
                raise ValueError("Cache 'key' must be provided as a string for 'set' operation.")
            
            self._cache[key] = data
            logger.info(f"CacheManagerNode: Data successfully set for key '{key}'.")
            return data
        
        elif operation == 'get':
            if not isinstance(key, str):
                logger.error("CacheManagerNode: 'key' must be a string for 'get' operation.")
                raise ValueError("Cache 'key' must be provided as a string for 'get' operation.")
            
            if key in self._cache:
                cached_data = self._cache[key]
                logger.debug(f"CacheManagerNode: Retrieved data for key '{key}'.")
                return cached_data
            else:
                logger.info(f"CacheManagerNode: No data found for key '{key}'. Returning default_value: {default_value}.")
                return default_value
        
        elif operation == 'clear':
            if key is None:
                initial_size = len(self._cache)
                self._cache.clear()
                logger.info(f"CacheManagerNode: Entire cache cleared. {initial_size} items removed.")
                return True
            else:
                if not isinstance(key, str):
                    logger.error("CacheManagerNode: 'key' must be a string for specific 'clear' operation.")
                    raise ValueError("Cache 'key' must be provided as a string for specific 'clear' operation.")
                
                if key in self._cache:
                    del self._cache[key]
                    logger.info(f"CacheManagerNode: Key '{key}' removed from cache.")
                    return True
                else:
                    logger.warning(f"CacheManagerNode: Attempted to clear non-existent key '{key}'.")
                    return False
        else:
            logger.error(f"CacheManagerNode: Invalid operation '{operation}' specified in context.")
            raise ValueError(f"Invalid cache operation: '{operation}'. Must be 'get', 'set', or 'clear'.")

