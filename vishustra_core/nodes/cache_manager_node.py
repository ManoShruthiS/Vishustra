import logging
import time
from typing import Any, Dict, Optional

# In a deployed Vishustra environment, vishustra_core would be installed
# and its modules directly importable. For isolated development or testing
# outside a full Vishustra setup, we provide a placeholder BaseNode.
try:
    from vishustra_core.nodes.base_node import BaseNode
except ImportError:
    # Placeholder BaseNode for environments where vishustra_core is not present.
    # This allows the file to be linted and basic syntax checked without the full framework.
    class BaseNode:
        """
        A placeholder BaseNode class for environments where vishustra_core is not
        directly available during isolated file development.
        In a production Vishustra setup, this would be imported from the framework.
        """
        @property
        def node_name(self) -> str:
            """Returns the descriptive name of the node."""
            raise NotImplementedError("Subclasses must implement 'node_name' property.")

        def process(self, data: Any, context: Dict[str, Any]) -> Any:
            """Processes the input data and context."""
            raise NotImplementedError("Subclasses must implement the 'process' method.")
    logging.warning("vishustra_core.nodes.base_node not found. Using a placeholder BaseNode. "
                    "Ensure vishustra_core is installed for production deployment.")

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    The CacheManagerNode provides essential caching capabilities for Vishustra pipelines.
    It supports 'set', 'get', and 'delete' operations on an in-memory cache,
    including Time-To-Live (TTL) functionality for cached items.

    This node is designed to manage transient data efficiently within a workflow,
    reducing redundant computations or external API calls by storing and retrieving
    intermediate results.

    Context Parameters:
    - 'cache_key' (str): The unique identifier for the cached item. (Required for all operations)
    - 'cache_operation' (str): The desired cache operation ('set', 'get', 'delete'). (Required)
    - 'cache_ttl' (int, optional): Time-To-Live in seconds for 'set' operations.
                                   If not provided, the item will persist until explicitly deleted
                                   or application restart. Must be a positive integer.
    - 'cache_data' (Any, optional): The specific data to be cached for 'set' operations.
                                    If provided, this takes precedence over the `data` argument
                                    passed to the `process` method.

    Returns:
    - For 'set': The data that was successfully cached.
    - For 'get': The cached data if found and not expired, otherwise `None`.
    - For 'delete': A boolean indicating if the item was successfully deleted.
    - Errors are logged, and a `ValueError` or generic `Exception` may be raised for critical failures.
    """

    def __init__(self):
        super().__init__()
        # In-memory dictionary to simulate a cache store.
        # Each entry stores {'value': Any, 'expiry': Optional[float]}.
        # In a real-world, highly distributed system, this would integrate
        # with external caching solutions like Redis, Memcached, etc.
        self._cache: Dict[str, Dict[str, Any]] = {}
        logger.info("CacheManagerNode initialized with an in-memory cache.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "CacheManager"

    def _is_expired(self, key: str) -> bool:
        """
        Checks if a cached item associated with the given key has expired.
        If expired, it removes the item from the cache.

        Args:
            key (str): The cache key to check.

        Returns:
            bool: True if the item is expired or not found, False otherwise.
        """
        entry = self._cache.get(key)
        if not entry:
            # Item not found in cache, treat as expired/non-existent
            return True

        expiry_time = entry.get('expiry')
        if expiry_time is None:
            # No expiry time means it never expires
            return False

        if time.time() >= expiry_time:
            logger.debug(f"Cache key '{key}' has expired. Removing from cache.")
            self._cache.pop(key, None)  # Remove expired item
            return True
        return False

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the data based on the specified cache operation ('set', 'get', 'delete').

        Args:
            data (Any): The primary input data. For 'set' operations, this is the data
                        to be cached if 'cache_data' is not explicitly provided in the context.
            context (Dict[str, Any]): A dictionary containing operation parameters
                                      such as 'cache_key', 'cache_operation', and 'cache_ttl'.

        Returns:
            Any: The result of the cache operation (e.g., cached data, success status).

        Raises:
            ValueError: If required context parameters are missing or invalid.
            Exception: For unexpected errors during cache operations.
        """
        cache_key: Optional[str] = context.get('cache_key')
        cache_operation: Optional[str] = context.get('cache_operation')
        cache_ttl: Optional[int] = context.get('cache_ttl')
        cache_data_from_context: Optional[Any] = context.get('cache_data')

        if not cache_key:
            logger.error("CacheManagerNode: 'cache_key' is a required parameter in context for any operation.")
            raise ValueError("Missing 'cache_key' in context for CacheManagerNode operation.")
        if not cache_operation:
            logger.error("CacheManagerNode: 'cache_operation' is a required parameter in context.")
            raise ValueError("Missing 'cache_operation' in context for CacheManagerNode.")

        # Normalize operation string for case-insensitivity
        cache_operation = cache_operation.lower()

        try:
            if cache_operation == 'set':
                # Determine the value to cache: context['cache_data'] takes precedence over `data` arg
                value_to_cache = cache_data_from_context if cache_data_from_context is not None else data
                
                expiry: Optional[float] = None
                if cache_ttl is not None:
                    if not isinstance(cache_ttl, int) or cache_ttl <= 0:
                        logger.warning(
                            f"Invalid 'cache_ttl' provided for key '{cache_key}': {cache_ttl}. "
                            "TTL must be a positive integer. Storing without expiration."
                        )
                        expiry = None
                    else:
                        expiry = time.time() + cache_ttl

                self._cache[cache_key] = {'value': value_to_cache, 'expiry': expiry}
                log_message = f"Cache key '{cache_key}' set successfully."
                if expiry:
                    log_message += f" Expires in {cache_ttl} seconds."
                else:
                    log_message += " No expiration set."
                logger.info(log_message)
                return value_to_cache

            elif cache_operation == 'get':
                if self._is_expired(cache_key):
                    logger.debug(f"Cache key '{cache_key}' not found or has expired. Returning None.")
                    return None
                
                cached_entry = self._cache.get(cache_key)
                if cached_entry:
                    logger.debug(f"Cache key '{cache_key}' retrieved successfully.")
                    return cached_entry['value']
                else:
                    # This path should ideally be rare given _is_expired, but defensive programming
                    logger.warning(f"Cache key '{cache_key}' unexpectedly not found in cache after "
                                   "expiry check. Possible concurrency issue or external modification.")
                    return None

            elif cache_operation == 'delete':
                if cache_key in self._cache:
                    del self._cache[cache_key]
                    logger.info(f"Cache key '{cache_key}' deleted successfully.")
                    return True
                else:
                    logger.debug(f"Attempted to delete non-existent cache key '{cache_key}'. No action taken.")
                    return False

            else:
                logger.error(
                    f"CacheManagerNode: Invalid 'cache_operation' specified: '{cache_operation}'. "
                    "Expected 'set', 'get', or 'delete'."
                )
                raise ValueError(f"Invalid 'cache_operation': {cache_operation}")

        except ValueError:
            # Re-raise ValueError from within the try block
            raise
        except Exception as e:
            logger.exception(f"An unexpected error occurred during cache operation '{cache_operation}' "
                             f"for key '{cache_key}': {e}")
            raise