import logging
from typing import Any, Dict, Optional

# Assuming BaseNode is located here as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A Vishustra node designed to manage a shared, in-memory cache within the
    orchestration context. This node enables other parts of the workflow to
    efficiently store, retrieve, update, and delete cached data.

    The cache itself is a standard Python dictionary, stored and managed under
    the key `_vishustra_cache_store` within the shared `context` dictionary
    provided to the `process` method. This allows the cache state to persist
    and be accessible across multiple nodes in an orchestration.

    Supported operations via the 'action' field in the input 'data':
    - 'set': Stores a value associated with a given key.
    - 'get': Retrieves the value associated with a given key.
    - 'delete': Removes a key-value pair from the cache.
    - 'clear': Empties the entire cache.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this cache manager node."""
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes various cache operations based on the provided `data` and
        interacts with the shared cache in the `context`.

        The `data` input is expected to be a dictionary specifying the desired
        cache operation and its relevant arguments:
        - For 'set': `{'action': 'set', 'key': str, 'value': Any}`
        - For 'get': `{'action': 'get', 'key': str}`
        - For 'delete': `{'action': 'delete', 'key': str}`
        - For 'clear': `{'action': 'clear'}`

        Args:
            data: A dictionary containing the 'action' and operation-specific parameters.
                  Must be a dictionary.
            context: The shared orchestration context, where the cache is managed
                     under the key `_vishustra_cache_store`.

        Returns:
            - For 'get' action: The cached value if found, otherwise `None`.
            - For 'set', 'delete', 'clear' actions: `None` (operations do not return data directly).

        Raises:
            ValueError: If `data` is malformed, missing required keys for an action,
                        or contains an unknown action.
            TypeError: If the cache store found in `context['_vishustra_cache_store']`
                       is not a dictionary.
        """
        if not isinstance(data, dict):
            logger.error(
                "CacheManagerNode received non-dictionary data. Expected a dict, got: %s (type: %s)",
                data, type(data)
            )
            raise ValueError(
                f"CacheManagerNode expects 'data' to be a dictionary for operations, "
                f"but received type: {type(data)}"
            )

        action: Optional[str] = data.get('action')
        cache_key: Optional[str] = data.get('key')
        value_to_cache: Any = data.get('value') # This can be None for get/delete/clear

        # Initialize or validate the cache store within the context
        # Using a prefixed key to avoid collisions with user-defined context variables
        cache_store = context.get('_vishustra_cache_store')

        if cache_store is None:
            context['_vishustra_cache_store'] = {}
            cache_store = context['_vishustra_cache_store']
            logger.debug("Initialized an empty cache store at context['_vishustra_cache_store'].")
        elif not isinstance(cache_store, dict):
            logger.error(
                "Cache store at context['_vishustra_cache_store'] is not a dictionary. "
                "Found type: %s", type(cache_store)
            )
            raise TypeError(
                f"Expected the cache store in context['_vishustra_cache_store'] to be a "
                f"dictionary, but found type: {type(cache_store)}."
            )

        if action == 'set':
            if not isinstance(cache_key, str) or not cache_key: # Ensure key is a non-empty string
                logger.error("Cache 'set' operation requires a valid, non-empty string 'key'. Received: %s", cache_key)
                raise ValueError("Cache 'set' operation requires a valid, non-empty string 'key'.")
            
            cache_store[cache_key] = value_to_cache
            logger.info("CacheManagerNode: Successfully set key '%s'.", cache_key)
            return None

        elif action == 'get':
            if not isinstance(cache_key, str) or not cache_key:
                logger.error("Cache 'get' operation requires a valid, non-empty string 'key'. Received: %s", cache_key)
                raise ValueError("Cache 'get' operation requires a valid, non-empty string 'key'.")
            
            cached_value = cache_store.get(cache_key)
            if cached_value is not None:
                logger.debug("CacheManagerNode: Successfully retrieved key '%s' from cache.", cache_key)
            else:
                logger.debug("CacheManagerNode: Key '%s' not found in cache.", cache_key)
            return cached_value

        elif action == 'delete':
            if not isinstance(cache_key, str) or not cache_key:
                logger.error("Cache 'delete' operation requires a valid, non-empty string 'key'. Received: %s", cache_key)
                raise ValueError("Cache 'delete' operation requires a valid, non-empty string 'key'.")
            
            if cache_key in cache_store:
                del cache_store[cache_key]
                logger.info("CacheManagerNode: Successfully deleted key '%s'.", cache_key)
            else:
                logger.debug("CacheManagerNode: Attempted to delete non-existent key '%s'. No action taken.", cache_key)
            return None

        elif action == 'clear':
            cache_store.clear()
            logger.info("CacheManagerNode: Successfully cleared all items from the cache.")
            return None

        else:
            logger.error("Unknown or missing cache 'action' in data: '%s'. Full data: %s", action, data)
            raise ValueError(
                f"Unknown or missing cache 'action': '{action}'. Expected 'get', 'set', 'delete', or 'clear'."
            )