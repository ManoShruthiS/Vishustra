import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class CacheManagerNode(BaseNode):
    """
    A Vishustra processing node designed for flexible cache operations (get, set, delete).

    This node interacts with a cache instance provided in the `context` dictionary
    under the key 'cache'. The node determines the specific cache action to perform
    based on the 'cache_action' key in the `context` (e.g., 'get', 'set', 'delete').

    For 'set' operations, the value to be stored must be present in `context['value_to_cache']`.
    The `data` input to the `process` method is consistently used as the cache key
    for all supported actions.

    The cache instance is expected to implement callable methods corresponding to
    the requested actions (e.g., `get(key)`, `set(key, value)`, `delete(key)`).

    Example context for a 'get' operation:
    ```python
    {'cache': my_cache_instance, 'cache_action': 'get'}
    ```

    Example context for a 'set' operation:
    ```python
    {'cache': my_cache_instance, 'cache_action': 'set', 'value_to_cache': 'some_value'}
    ```

    Example context for a 'delete' operation:
    ```python
    {'cache': my_cache_instance, 'cache_action': 'delete'}
    ```
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes a specified cache operation (get, set, delete) using a provided cache instance.

        Args:
            data: The key for the cache operation. This can be any hashable type.
            context: A dictionary containing:
                - 'cache': The cache instance. It must expose callable methods
                           for 'get', 'set', or 'delete' depending on the action.
                - 'cache_action': A string specifying the action ('get', 'set', 'delete').
                - 'value_to_cache' (optional, required for 'set' action):
                  The value to be stored in the cache.

        Returns:
            - For 'get' action: The cached value associated with the key, or `None` if not found.
            - For 'set' action: `True` if the value was successfully stored.
            - For 'delete' action: `True` if the key was successfully removed (or if it was not present).

        Raises:
            ValueError: If critical context parameters ('cache', 'cache_action', 'value_to_cache' for 'set')
                        are missing or if an unknown `cache_action` is specified.
            TypeError: If the 'cache' object does not implement the required callable method
                       for the specified action.
            Exception: Any exception raised by the underlying cache instance during its operation.
        """
        cache_key = data

        if 'cache' not in context:
            logger.error("CacheManagerNode: Context missing 'cache' instance.")
            raise ValueError("Missing 'cache' instance in context for CacheManagerNode.")

        cache_instance = context['cache']

        if 'cache_action' not in context:
            logger.error("CacheManagerNode: Context missing 'cache_action'.")
            raise ValueError("Missing 'cache_action' in context for CacheManagerNode.")

        action = str(context['cache_action']).lower()
        logger.debug(f"CacheManagerNode initiated with action '{action}' for key '{cache_key}'.")

        try:
            if action == 'get':
                if not callable(getattr(cache_instance, 'get', None)):
                    logger.error(f"Cache instance ({type(cache_instance)}) does not implement a callable 'get' method.")
                    raise TypeError("Cache instance does not implement a callable 'get' method for 'get' action.")
                
                result = cache_instance.get(cache_key)
                if result is not None:
                    logger.info(f"Successfully retrieved item for key '{cache_key}' from cache.")
                else:
                    logger.debug(f"Item not found for key '{cache_key}' in cache.")
                return result

            elif action == 'set':
                if not callable(getattr(cache_instance, 'set', None)):
                    logger.error(f"Cache instance ({type(cache_instance)}) does not implement a callable 'set' method.")
                    raise TypeError("Cache instance does not implement a callable 'set' method for 'set' action.")
                if 'value_to_cache' not in context:
                    logger.error(f"CacheManagerNode: Context missing 'value_to_cache' for 'set' action with key '{cache_key}'.")
                    raise ValueError("Missing 'value_to_cache' in context for 'set' action.")
                
                value_to_cache = context['value_to_cache']
                cache_instance.set(cache_key, value_to_cache)
                logger.info(f"Successfully set item for key '{cache_key}' in cache.")
                return True

            elif action == 'delete':
                if not callable(getattr(cache_instance, 'delete', None)):
                    logger.error(f"Cache instance ({type(cache_instance)}) does not implement a callable 'delete' method.")
                    raise TypeError("Cache instance does not implement a callable 'delete' method for 'delete' action.")
                
                cache_instance.delete(cache_key)
                logger.info(f"Successfully deleted item for key '{cache_key}' from cache (if present).")
                return True

            else:
                logger.error(f"CacheManagerNode: Unknown cache action '{action}' provided.")
                raise ValueError(f"Unknown cache action: '{action}'. Expected 'get', 'set', or 'delete'.")

        except (ValueError, TypeError) as e:
            # Re-raise explicit validation errors without wrapping, as they are already specific
            raise e
        except Exception as e:
            logger.exception(f"An unexpected error occurred during cache operation '{action}' for key '{cache_key}': {e}")
            raise # Re-raise any other exceptions from the cache instance