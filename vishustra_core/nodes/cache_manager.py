import logging
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A processing node designed to manage various cache operations (get, set, delete, has)
    on a specified cache instance.

    This node acts as an interface to a cache backend, allowing flexible interaction
    within an orchestration flow. It expects the cache instance and desired operation
    to be provided via the `context` dictionary.

    Expected `context` keys for the `process` method:
    - 'cache_instance': (Required) The cache object. This can be a simple dictionary
                        or a custom cache class that implements methods like `get`,
                        `set`, `delete`, and supports `__contains__` or a `has` method.
    - 'cache_operation': (Required) A string indicating the specific cache operation:
                         'get', 'set', 'delete', or 'has'. Case-insensitive.
    - 'cache_value': (Required for 'set' operation) The value to be stored in the cache.
    - 'cache_ttl': (Optional for 'set' operation) Time-to-live (in seconds) for the
                   cached item. The actual handling of TTL depends on the capabilities
                   of the `cache_instance` implementation.

    The `data` parameter to the `process` method is used as the cache key for the operation.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes a cache operation based on the provided data (cache key) and context.

        Args:
            data: The cache key for the operation.
            context: A dictionary containing the cache instance, the operation type,
                     and potentially the value to store and its TTL.

        Returns:
            The result of the cache operation:
            - For 'get': The cached value if found; otherwise, None.
            - For 'set': The value that was successfully stored.
            - For 'delete': True if the key was deleted; False if the key was not found.
            - For 'has': True if the key exists in the cache; False otherwise.

        Raises:
            ValueError: If 'cache_instance' or 'cache_operation' are missing
                        in the context, if 'cache_value' is missing for a 'set' operation,
                        or if an unknown operation is specified.
            TypeError: If the 'cache_instance' provided does not support the required methods
                       or protocol for the specified operation.
            Exception: Propagates any other exceptions originating from the cache instance.
        """
        cache_key = data
        cache_instance: Optional[Any] = context.get('cache_instance')
        cache_operation: Optional[str] = context.get('cache_operation')

        if cache_instance is None:
            logger.error("Context missing 'cache_instance' for CacheManagerNode. Cannot perform cache operation.")
            raise ValueError("Required 'cache_instance' not found in context.")
        if not isinstance(cache_instance, (dict, object)):
             logger.error(f"Invalid 'cache_instance' type provided: {type(cache_instance)}. Expected a dict-like object or an object with cache methods.")
             raise TypeError(f"Invalid 'cache_instance' type: {type(cache_instance)}. Expected dict-like or object with cache methods.")

        if cache_operation is None:
            logger.error("Context missing 'cache_operation' for CacheManagerNode. Cannot determine action.")
            raise ValueError("Required 'cache_operation' not found in context.")

        cache_operation = cache_operation.lower()
        logger.debug(f"CacheManagerNode initiated operation '{cache_operation}' for key '{cache_key}'.")

        try:
            if cache_operation == 'get':
                if hasattr(cache_instance, 'get') and callable(getattr(cache_instance, 'get')):
                    result = cache_instance.get(cache_key)
                elif isinstance(cache_instance, dict):
                    result = cache_instance.get(cache_key)
                else:
                    logger.error(f"Cache instance of type {type(cache_instance)} does not support a 'get' method for 'get' operation.")
                    raise TypeError(f"Cache instance of type {type(cache_instance)} does not support a 'get' method for 'get' operation.")
                logger.debug(f"Cache 'get' operation for key '{cache_key}': {'hit' if result is not None else 'miss'}")
                return result

            elif cache_operation == 'set':
                cache_value = context.get('cache_value')
                if cache_value is None:
                    logger.error(f"Context missing 'cache_value' for 'set' operation on key '{cache_key}'.")
                    raise ValueError(f"Required 'cache_value' not found in context for 'set' operation on key '{cache_key}'.")

                cache_ttl = context.get('cache_ttl')

                if hasattr(cache_instance, 'set') and callable(getattr(cache_instance, 'set')):
                    try:
                        if cache_ttl is not None:
                            cache_instance.set(cache_key, cache_value, ttl=cache_ttl)
                        else:
                            cache_instance.set(cache_key, cache_value)
                    except TypeError as e: # Handle cases where custom 'set' might not accept 'ttl'
                         if cache_ttl is not None:
                            logger.warning(f"Cache instance 'set' method for {type(cache_instance)} raised TypeError (likely 'ttl' argument not supported): {e}. Storing '{cache_key}' without TTL.")
                            cache_instance.set(cache_key, cache_value) # Try without TTL
                         else:
                            raise # Re-raise if TypeError for other reasons
                elif isinstance(cache_instance, dict):
                    cache_instance[cache_key] = cache_value
                else:
                    logger.error(f"Cache instance of type {type(cache_instance)} does not support a 'set' method or dictionary assignment for 'set' operation.")
                    raise TypeError(f"Cache instance of type {type(cache_instance)} does not support a 'set' method or dictionary assignment for 'set' operation.")
                logger.debug(f"Cache 'set' operation for key '{cache_key}'. Value stored.")
                return cache_value

            elif cache_operation == 'delete':
                if hasattr(cache_instance, 'delete') and callable(getattr(cache_instance, 'delete')):
                    result = cache_instance.delete(cache_key) # Assume custom delete returns success boolean
                elif isinstance(cache_instance, dict):
                    if cache_key in cache_instance:
                        del cache_instance[cache_key]
                        result = True
                    else:
                        result = False
                else:
                    logger.error(f"Cache instance of type {type(cache_instance)} does not support a 'delete' method or dictionary item deletion for 'delete' operation.")
                    raise TypeError(f"Cache instance of type {type(cache_instance)} does not support a 'delete' method or dictionary item deletion for 'delete' operation.")

                logger.debug(f"Cache 'delete' operation for key '{cache_key}'. Success: {result}")
                return result

            elif cache_operation == 'has':
                if hasattr(cache_instance, 'has') and callable(getattr(cache_instance, 'has')):
                    result = cache_instance.has(cache_key)
                elif hasattr(cache_instance, '__contains__') and callable(getattr(cache_instance, '__contains__')):
                    result = cache_key in cache_instance
                else:
                    logger.error(f"Cache instance of type {type(cache_instance)} does not support a 'has' method or '__contains__' for 'has' operation.")
                    raise TypeError(f"Cache instance of type {type(cache_instance)} does not support a 'has' method or '__contains__' for 'has' operation.")

                logger.debug(f"Cache 'has' operation for key '{cache_key}': {result}")
                return result

            else:
                logger.error(f"Unknown cache operation '{cache_operation}' requested for CacheManagerNode.")
                raise ValueError(f"Unknown cache operation: {cache_operation}")

        except Exception as e:
            logger.exception(f"An unexpected error occurred during cache operation '{cache_operation}' for key '{cache_key}': {e}")
            raise # Re-raise the exception after logging for upstream handling
