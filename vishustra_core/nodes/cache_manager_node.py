import logging
from typing import Any, Dict, MutableMapping # MutableMapping for more precise type hint for cache_storage

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A Vishustra processing node designed for managing a cache within an orchestration flow.

    This node supports common cache operations like 'get', 'set', 'clear_key', and 'clear_all',
    controlled by parameters provided in the `context` dictionary. It expects a mutable
    dictionary-like object for cache storage, allowing for flexible integration with various
    caching mechanisms (e.g., in-memory dict, an LRU cache, or even a client for external
    caches like Redis if adapted by a custom cache_storage object).

    Expected context parameters:
    - 'cache_storage' (MutableMapping[Any, Any]): The mutable dictionary-like object serving as the cache.
                                                 This is a mandatory parameter.
    - 'cache_action' (str): The desired cache operation. Must be one of:
                            'get': Retrieve a value from the cache.
                            'set': Store a value in the cache.
                            'clear_key': Remove a specific key-value pair from the cache.
                            'clear_all': Clear all entries from the cache.
                                         This is a mandatory parameter.
    - 'cache_value_to_set' (Any, optional): The value to be stored when 'cache_action' is 'set'.
                                            If not provided, the input 'data' to the process method
                                            will be used as the value.

    Input 'data' for process method:
    - For 'get', 'set', 'clear_key' actions: The 'data' argument is treated as the cache key.
    - For 'clear_all' action: The 'data' argument is ignored, as the operation is global.

    Output of process method:
    - 'get': The cached value if found for the given key, otherwise None.
    - 'set': The value that was successfully stored in the cache.
    - 'clear_key': True if the key was found and removed, False otherwise.
    - 'clear_all': True after the cache has been cleared successfully.

    Raises:
        ValueError: If 'cache_storage' or 'cache_action' is missing in context,
                    or if an unsupported 'cache_action' is provided.
        TypeError: If 'cache_storage' is not a mutable dictionary-like object (e.g., dict).
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "CacheManagerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes a cache operation based on the 'cache_action' specified in the context.

        Args:
            data (Any): The primary input data for the node. This typically serves as the
                        cache key for 'get', 'set', and 'clear_key' operations.
            context (Dict[str, Any]): A dictionary containing parameters for the cache operation,
                                       including 'cache_storage' and 'cache_action'.

        Returns:
            Any: The result of the cache operation (e.g., retrieved value, status boolean).

        Raises:
            ValueError: If required context parameters are missing or invalid.
            TypeError: If 'cache_storage' is not a mutable dictionary-like object.
        """
        # 1. Validate and retrieve cache storage
        cache_storage: Optional[MutableMapping[Any, Any]] = context.get('cache_storage')
        if not isinstance(cache_storage, MutableMapping):
            logger.error(
                f"[{self.node_name}] 'cache_storage' not found or is not a mutable mapping "
                f"in context for node {self.node_name}. Got type: {type(cache_storage)}"
            )
            raise TypeError("Context parameter 'cache_storage' must be a mutable dictionary-like object.")

        # 2. Validate and retrieve cache action
        action: Optional[str] = context.get('cache_action')
        if not isinstance(action, str):
            logger.error(
                f"[{self.node_name}] 'cache_action' not found or is not a string "
                f"in context for node {self.node_name}. Got type: {type(action)}"
            )
            raise ValueError("Context parameter 'cache_action' must be a string.")

        logger.debug(f"[{self.node_name}] Processing with action '{action}' for data: {data!r}")

        if action == 'get':
            cache_key = data
            result = cache_storage.get(cache_key)
            if result is not None:
                logger.info(f"[{self.node_name}] Cache hit for key: {cache_key!r}")
            else:
                logger.info(f"[{self.node_name}] Cache miss for key: {cache_key!r}")
            return result
        elif action == 'set':
            cache_key = data
            # Use 'cache_value_to_set' from context if provided, otherwise use the input 'data' itself.
            value_to_set = context.get('cache_value_to_set', data)
            cache_storage[cache_key] = value_to_set
            logger.info(f"[{self.node_name}] Set cache key {cache_key!r} with value: {value_to_set!r}")
            return value_to_set
        elif action == 'clear_key':
            key_to_clear = data
            if key_to_clear in cache_storage:
                del cache_storage[key_to_clear]
                logger.info(f"[{self.node_name}] Cleared cache entry for key: {key_to_clear!r}")
                return True
            else:
                logger.info(f"[{self.node_name}] Key {key_to_clear!r} not found in cache for clearing.")
                return False
        elif action == 'clear_all':
            cache_storage.clear()
            logger.info(f"[{self.node_name}] Cleared all entries from cache.")
            return True
        else:
            logger.error(
                f"[{self.node_name}] Unsupported cache_action '{action}' provided in context "
                f"for node {self.node_name}."
            )
            raise ValueError(f"Unsupported cache_action: '{action}'")