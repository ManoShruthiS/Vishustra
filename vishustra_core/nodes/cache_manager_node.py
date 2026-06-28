import logging
from typing import Any, Dict, Optional
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheNodeError(Exception):
    """Custom exception for errors specific to the CacheManagerNode."""
    pass

class CacheManagerNode(BaseNode):
    """
    A Vishustra processing node designed for managing data caching operations.

    This node provides functionality to either retrieve ('get') or store ('set')
    data within a shared cache store, enabling efficient reuse of computation
    results across an orchestration.

    The node's behavior is configured via the `context` dictionary passed to
    its `process` method.

    Context expectations for `process` method:
    - 'cache_store': (Required) A mutable dictionary-like object (e.g., dict, LRUCache instance)
                     that will serve as the underlying caching mechanism.
    - 'cache_key': (Required) The unique identifier or key under which data is to be
                   stored or retrieved from the `cache_store`.
    - 'cache_operation': (Required) A string specifying the desired cache action.
                         Valid values are:
                         - 'get': Attempts to retrieve data associated with 'cache_key'.
                                  The input `data` to the `process` method is ignored for this operation.
                         - 'set': Stores the input `data` received by the `process` method
                                  under the specified 'cache_key'.
    """

    @property
    def node_name(self) -> str:
        """Returns the programmatic name of the node."""
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes the specified cache operation ('get' or 'set') using the
        provided cache store and key.

        Args:
            data: For a 'set' operation, this is the value to be cached.
                  For a 'get' operation, this input is disregarded.
            context: A dictionary containing operational parameters for the node.
                     Must include 'cache_store', 'cache_key', and 'cache_operation'.

        Returns:
            Any:
                - For a 'get' operation: Returns the cached value if found,
                  otherwise returns `None` to indicate a cache miss.
                - For a 'set' operation: Returns the data that was successfully
                  stored in the cache, allowing for result chaining.

        Raises:
            CacheNodeError: If critical context keys ('cache_store', 'cache_key',
                            'cache_operation') are missing or invalid, or if an
                            unsupported 'cache_operation' is specified.
        """
        try:
            cache_store = context.get('cache_store')
            cache_key = context.get('cache_key')
            cache_operation = context.get('cache_operation')

            if cache_store is None:
                logger.error("Context for CacheManagerNode is missing the 'cache_store' key.")
                raise CacheNodeError("Missing 'cache_store' in context. A cache object is required.")
            # Basic check for dictionary-like behavior
            if not (hasattr(cache_store, '__getitem__') and
                    hasattr(cache_store, '__setitem__') and
                    hasattr(cache_store, '__contains__')):
                logger.error(f"Invalid 'cache_store' object type: {type(cache_store)}. Must be dictionary-like.")
                raise CacheNodeError(
                    f"Invalid 'cache_store' object. Expected dictionary-like (e.g., dict), got {type(cache_store)}."
                )

            if cache_key is None:
                logger.error("Context for CacheManagerNode is missing the 'cache_key' for the operation.")
                raise CacheNodeError("Missing 'cache_key' in context. A key is required for all cache operations.")
            
            if cache_operation is None:
                logger.error("Context for CacheManagerNode is missing the 'cache_operation'.")
                raise CacheNodeError("Missing 'cache_operation' in context. Specify either 'get' or 'set'.")

            if cache_operation == 'get':
                return self._handle_get(cache_store, cache_key)
            elif cache_operation == 'set':
                return self._handle_set(cache_store, cache_key, data)
            else:
                logger.error(f"Unsupported cache_operation: '{cache_operation}'. Expected 'get' or 'set'.")
                raise CacheNodeError(
                    f"Invalid 'cache_operation' value: '{cache_operation}'. Expected 'get' or 'set'."
                )

        except CacheNodeError:
            # Re-raise custom exceptions directly as they already contain context
            raise
        except Exception as e:
            # Catch any other unexpected errors and wrap them for consistent error reporting
            logger.exception(f"An unexpected error occurred in CacheManagerNode during processing for key '{cache_key}'.")
            raise CacheNodeError(f"Unexpected error during cache operation for key '{cache_key}': {e}") from e

    def _handle_get(self, cache_store: Any, cache_key: Any) -> Optional[Any]:
        """
        Internal handler for the 'get' cache operation.

        Args:
            cache_store: The cache object.
            cache_key: The key to retrieve.

        Returns:
            The cached value if found, otherwise `None`.
        """
        if cache_key in cache_store:
            value = cache_store[cache_key]
            logger.debug(f"Cache hit for key: '{cache_key}'.")
            return value
        else:
            logger.debug(f"Cache miss for key: '{cache_key}'.")
            return None

    def _handle_set(self, cache_store: Any, cache_key: Any, value: Any) -> Any:
        """
        Internal handler for the 'set' cache operation.

        Args:
            cache_store: The cache object.
            cache_key: The key under which to store the value.
            value: The data to be cached.

        Returns:
            The value that was just stored in the cache.
        """
        cache_store[cache_key] = value
        logger.debug(f"Data successfully set in cache for key: '{cache_key}'.")
        return value # Return the value, enabling potential chaining or logging.
