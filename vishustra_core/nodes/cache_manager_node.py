import logging
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManager(BaseNode):
    """
    A Vishustra node that provides in-memory caching capabilities.

    This node supports 'get', 'set', and 'delete' operations for managing
    cached data. It expects input data to be a dictionary specifying the
    operation, key, and optionally the value.

    Input Data Format (data parameter in process method):
    - To get an item:    {"operation": "get", "key": "your_key"}
    - To set an item:    {"operation": "set", "key": "your_key", "value": "your_value"}
    - To delete an item: {"operation": "delete", "key": "your_key"}

    Output:
    - For 'get': The cached value if found, otherwise None. The context will
                 include 'cache_hit': True/False.
    - For 'set': A dictionary like {"status": "success", "operation": "set", "key": "your_key"}
    - For 'delete': A dictionary like {"status": "success", "operation": "delete", "key": "your_key"}
                  or {"status": "not_found", ...} if the key wasn't present.
    """

    _cache: Dict[str, Any]

    def __init__(self, initial_cache: Optional[Dict[str, Any]] = None):
        """
        Initializes the CacheManager node.

        Args:
            initial_cache (Optional[Dict[str, Any]]): An optional dictionary
                                                       to pre-populate the cache.
        """
        self._cache = initial_cache if initial_cache is not None else {}
        logger.debug(f"CacheManager initialized with {len(self._cache)} initial items.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to perform cache operations (get, set, delete).

        Args:
            data (Any): A dictionary containing the 'operation', 'key', and
                        optionally 'value' for the cache.
            context (Dict[str, Any]): A dictionary for shared context information.
                                      May be updated with 'cache_hit' status.

        Returns:
            Any: The result of the cache operation (e.g., cached value, status dict).

        Raises:
            ValueError: If the input data format is incorrect or missing required fields.
            RuntimeError: If an unexpected error occurs during a cache operation.
        """
        if not isinstance(data, dict):
            logger.error("CacheManager expects 'data' to be a dictionary.")
            raise ValueError("Invalid input data format: expected a dictionary.")

        operation = data.get("operation")
        key = data.get("key")
        value = data.get("value")

        if not operation or not key:
            logger.error(f"Missing 'operation' or 'key' in input data: {data}")
            raise ValueError("Input data must contain 'operation' and 'key'.")

        try:
            if operation == "get":
                if key in self._cache:
                    logger.debug(f"Cache hit for key: '{key}'")
                    context["cache_hit"] = True
                    return self._cache[key]
                else:
                    logger.debug(f"Cache miss for key: '{key}'")
                    context["cache_hit"] = False
                    return None
            elif operation == "set":
                if value is None:
                    logger.error(f"Missing 'value' for 'set' operation with key: '{key}'")
                    raise ValueError("Value must be provided for 'set' operation.")
                self._cache[key] = value
                logger.info(f"Set cache entry for key: '{key}'")
                return {"status": "success", "operation": "set", "key": key}
            elif operation == "delete":
                if key in self._cache:
                    del self._cache[key]
                    logger.info(f"Deleted cache entry for key: '{key}'")
                    return {"status": "success", "operation": "delete", "key": key}
                else:
                    logger.warning(f"Attempted to delete non-existent cache key: '{key}'")
                    return {"status": "not_found", "operation": "delete", "key": key}
            else:
                logger.error(f"Unsupported cache operation: '{operation}'")
                raise ValueError(f"Unsupported cache operation: '{operation}'. "
                                 f"Supported operations are 'get', 'set', 'delete'.")
        except Exception as e:
            logger.exception(f"An error occurred during cache operation '{operation}' for key '{key}': {e}")
            raise RuntimeError(f"Cache operation failed for key '{key}'. "
                               f"Original error: {e}") from e