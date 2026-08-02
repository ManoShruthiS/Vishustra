import logging
from typing import Any, Dict, Optional

# Assuming this path is correct based on the project context's BaseNode definition
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A Vishustra node designed to manage cache interactions within an orchestration flow.

    This node facilitates common cache operations such as 'get', 'set', and 'invalidate'
    by interacting with a cache client provided in the context.

    The 'data' input for the 'process' method should be a dictionary specifying the
    desired cache operation and its parameters. The 'context' dictionary must
    contain the 'cache_client' instance, which should expose 'get', 'set', and
    'delete' (or 'remove') methods.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes a cache operation based on the provided data and available context.

        The expected structure for the 'data' argument is a dictionary:
        - For 'get' operation: `{"operation": "get", "key": "my_cache_key"}`
        - For 'set' operation: `{"operation": "set", "key": "my_cache_key", "value": "my_value"}`
        - For 'invalidate' operation: `{"operation": "invalidate", "key": "my_cache_key"}`

        The 'context' dictionary must contain:
        - 'cache_client' (Any): An object representing the cache client (e.g., a Redis client,
                                a simple dictionary, etc.) that implements `get`, `set`,
                                and `delete` (or `remove`) methods.

        Args:
            data (Any): A dictionary containing 'operation', 'key', and optionally 'value'.
            context (Dict[str, Any]): The execution context, including the 'cache_client'.

        Returns:
            Any:
                - For 'get': The value retrieved from the cache, or `None` if not found.
                - For 'set': The value that was successfully set in the cache.
                - For 'invalidate': The key that was invalidated.

        Raises:
            ValueError: If 'cache_client' is missing in context, or if 'operation'
                        or 'key' are missing/invalid in data, or if an unsupported
                        operation is requested.
            TypeError: If the 'cache_client' does not implement the necessary method
                       for the requested operation.
            RuntimeError: For any other unexpected errors during cache interaction.
        """
        cache_client = context.get('cache_client')
        if not cache_client:
            logger.error("CacheManagerNode: Missing 'cache_client' in the context.")
            raise ValueError("Required 'cache_client' not found in context.")

        if not isinstance(data, dict):
            logger.error(f"CacheManagerNode: Invalid data format. Expected dict, got {type(data)}.")
            raise ValueError("Input data must be a dictionary specifying the cache operation.")

        operation: Optional[str] = data.get('operation')
        key: Optional[str] = data.get('key')
        value: Any = data.get('value') # 'value' might be None or missing for get/invalidate

        if not operation or not isinstance(operation, str):
            logger.error(f"CacheManagerNode: Missing or invalid 'operation' in data: {data}")
            raise ValueError("Cache operation 'operation' (str) is required in data.")
        operation = operation.lower()

        if not key or not isinstance(key, str):
            logger.error(f"CacheManagerNode: Missing or invalid 'key' in data: {data}")
            raise ValueError("Cache key 'key' (str) is required in data.")

        try:
            if operation == 'get':
                if not hasattr(cache_client, 'get'):
                    raise TypeError("Cache client does not implement a 'get' method.")
                result = cache_client.get(key)
                logger.debug(f"CacheManagerNode: 'get' operation for key '{key}'. Found: {result is not None}")
                return result
            elif operation == 'set':
                # Check if 'value' key is present in data, allowing explicit None values
                if 'value' not in data:
                    logger.error(f"CacheManagerNode: 'value' is required for 'set' operation for key '{key}'. Data: {data}")
                    raise ValueError("Value must be provided for 'set' operation.")
                if not hasattr(cache_client, 'set'):
                    raise TypeError("Cache client does not implement a 'set' method.")
                cache_client.set(key, value)
                logger.debug(f"CacheManagerNode: 'set' operation for key '{key}' with value of type {type(value)}")
                return value
            elif operation == 'invalidate':
                if hasattr(cache_client, 'delete'):
                    cache_client.delete(key)
                elif hasattr(cache_client, 'remove'):
                    # Some clients use 'remove' instead of 'delete'
                    cache_client.remove(key)
                else:
                    raise TypeError("Cache client does not implement 'delete' or 'remove' for invalidation.")
                logger.debug(f"CacheManagerNode: 'invalidate' operation for key '{key}'")
                return key
            else:
                logger.error(f"CacheManagerNode: Unsupported cache operation requested: '{operation}'")
                raise ValueError(f"Unsupported cache operation: '{operation}'. Expected 'get', 'set', or 'invalidate'.")
        except (AttributeError, TypeError) as ate:
            logger.error(f"CacheManagerNode: Cache client interaction failed for operation '{operation}' on key '{key}': {ate}", exc_info=True)
            raise TypeError(f"Cache client method error for operation '{operation}': {ate}") from ate
        except Exception as e:
            logger.error(f"CacheManagerNode: An unexpected error occurred during cache operation '{operation}' for key '{key}': {e}", exc_info=True)
            raise RuntimeError(f"Cache operation failed for key '{key}': {e}") from e
