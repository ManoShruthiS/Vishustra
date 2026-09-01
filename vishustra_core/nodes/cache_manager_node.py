import logging
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A Vishustra node that provides in-memory cache management capabilities.

    This node supports 'get', 'set', and 'invalidate' operations on its internal
    dictionary-based cache. It allows orchestrators to store intermediate results
    or frequently accessed data.

    The `data` input to the `process` method serves as the value to be stored
    during a 'set' operation. For 'get' and 'invalidate', the `data` input
    is not directly utilized by the cache logic, as the operation relies
    on the `cache_key` provided in the `context`.

    Context Parameters (expected in `context` dictionary):
        - 'cache_operation' (str, required):
            Specifies the cache operation to perform: 'get', 'set', or 'invalidate'.
        - 'cache_key' (Any, required):
            The unique key to identify the cache entry.

    Returns from `process` method:
        - For 'get': The cached value if found. Returns `None` if the key
          does not exist in the cache.
        - For 'set': The value that was successfully stored in the cache.
        - For 'invalidate': `True` if the key was found and removed from the cache,
          `False` if the key was not found.

    Raises:
        ValueError: If essential `context` parameters (`cache_operation`, `cache_key`)
                    are missing or if an unknown operation is requested.
    """

    def __init__(self):
        """
        Initializes the in-memory cache for this node.
        """
        self._cache: Dict[Any, Any] = {}
        logger.debug(f"{self.node_name} initialized with an empty in-memory cache.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "CacheManagerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes the specified cache operation.

        Args:
            data (Any):
                For 'set' operations, this is the value to be stored in the cache.
                For 'get' and 'invalidate' operations, this input is generally
                ignored by the cache manager, as the key is specified in `context`.
            context (Dict[str, Any]):
                A dictionary providing control parameters for the cache operation.
                Must include 'cache_operation' and 'cache_key'.

        Returns:
            Any: The result of the cache operation. The type and meaning of the
                 return value vary based on the 'cache_operation' (see class docstring).

        Raises:
            ValueError: If 'cache_operation' or 'cache_key' are missing from context,
                        or if an unsupported 'cache_operation' is provided.
        """
        operation: Optional[str] = context.get('cache_operation')
        key: Any = context.get('cache_key')

        if operation is None:
            logger.error("Missing 'cache_operation' in context for CacheManagerNode.")
            raise ValueError("CacheManagerNode requires 'cache_operation' in context.")
        if key is None:
            logger.error(f"Missing 'cache_key' for operation '{operation}' in context for CacheManagerNode.")
            raise ValueError(f"CacheManagerNode requires 'cache_key' for operation '{operation}'.")

        logger.debug(f"{self.node_name} processing operation: '{operation}' for key: '{key}'.")

        if operation == 'get':
            cached_value = self._cache.get(key)
            if cached_value is not None or key in self._cache: # Differentiate None as value from missing key
                logger.info(f"Cache hit for key: '{key}'.")
                return cached_value
            else:
                logger.info(f"Cache miss for key: '{key}'.")
                return None
        elif operation == 'set':
            self._cache[key] = data
            logger.info(f"Value cached for key: '{key}'.")
            return data  # Return the value that was just set
        elif operation == 'invalidate':
            if key in self._cache:
                del self._cache[key]
                logger.info(f"Invalidated cache entry for key: '{key}'.")
                return True
            else:
                logger.info(f"Attempted to invalidate non-existent cache entry for key: '{key}'.")
                return False
        else:
            logger.error(f"Unknown cache operation: '{operation}' provided to CacheManagerNode.")
            raise ValueError(
                f"Unknown cache operation: '{operation}'. "
                "Supported operations are 'get', 'set', or 'invalidate'."
            )