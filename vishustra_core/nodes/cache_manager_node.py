import logging
from typing import Any, Dict, Callable, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A Vishustra processing node designed to manage data caching operations.

    This node provides functionality to:
    1.  **Lookup and Compute (default mode):** Retrieve a value from a cache based on a key.
        If the value is not found (cache miss), it can optionally use a provided
        `cache_miss_handler` to compute the value, store it in the cache, and then return it.
    2.  **Store:** Explicitly store a key-value pair into the cache.

    The mode of operation (`lookup` or `store`) is determined by the 'cache_action'
    key within the `context` dictionary.

    Context Requirements:
    -   `'cache_store'`: (REQUIRED) A mutable dictionary-like object that serves as the cache.
                          This should typically be shared across relevant nodes.
    -   `'cache_action'`: (OPTIONAL, defaults to 'lookup') Specifies the action: 'lookup' or 'store'.

    Specific to 'lookup' mode:
    -   Input `data`: The key (any hashable type) to look up in the cache.
    -   `'cache_miss_handler'`: (OPTIONAL) A callable `(key: Any, context: Dict[str, Any]) -> Any`
                                  function that will be invoked if a cache miss occurs.
                                  It should compute and return the desired value.
                                  If not provided, a cache miss will result in `None` being returned
                                  by this node, along with a warning log.

    Specific to 'store' mode:
    -   Input `data`: A dictionary `{'key': Any, 'value': Any}` containing the key
                      under which the value should be stored.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "CacheManagerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes the cache management operation based on the 'cache_action' in context.

        Args:
            data (Any):
                - For 'lookup' action: The cache key to retrieve.
                - For 'store' action: A dictionary `{'key': Any, 'value': Any}`.
            context (Dict[str, Any]):
                The operational context, containing:
                - 'cache_store': The dictionary-like cache object.
                - 'cache_action': The desired action ('lookup' or 'store').
                - 'cache_miss_handler': (Optional) Callable for cache misses.

        Returns:
            Any:
                - For 'lookup' action: The cached value, the resolved value from the handler,
                                       or `None` if not found and no handler.
                - For 'store' action: The value that was stored.

        Raises:
            ValueError: If 'cache_store' is missing or invalid, or if 'cache_action'
                        is unknown, or if 'data' is malformed for 'store' action.
            RuntimeError: If a 'cache_miss_handler' fails during execution.
        """
        cache_store = context.get('cache_store')
        if not isinstance(cache_store, dict):
            logger.error(f"[{self.node_name}] Context must provide a valid 'cache_store' (dict-like object). Found type: {type(cache_store)}")
            raise ValueError("Context must contain a 'cache_store' (dict-like object).")

        cache_action = context.get('cache_action', 'lookup')

        if cache_action == 'lookup':
            key = data
            # Basic check for hashability. Full check requires attempting to use it as a key.
            if not isinstance(key, (str, int, float, bool, tuple, frozenset, type(None))):
                logger.debug(f"[{self.node_name}] Attempting to lookup with potentially unhashable key type: {type(key)}. Proceeding with caution.")
            
            try:
                if key in cache_store:
                    logger.debug(f"[{self.node_name}] Cache hit for key: '{key}'")
                    return cache_store[key]
                else:
                    logger.debug(f"[{self.node_name}] Cache miss for key: '{key}'. Attempting to resolve via handler.")
                    cache_miss_handler: Optional[Callable[[Any, Dict[str, Any]], Any]] = context.get('cache_miss_handler')

                    if callable(cache_miss_handler):
                        try:
                            resolved_value = cache_miss_handler(key, context)
                            cache_store[key] = resolved_value
                            logger.info(f"[{self.node_name}] Resolved and cached value for key: '{key}'")
                            return resolved_value
                        except Exception as e:
                            logger.error(f"[{self.node_name}] Error calling 'cache_miss_handler' for key '{key}': {e}", exc_info=True)
                            raise RuntimeError(f"Failed to resolve value for key '{key}' via handler.") from e
                    else:
                        logger.warning(
                            f"[{self.node_name}] Cache miss for key '{key}' and no callable "
                            "'cache_miss_handler' provided in context. Returning None."
                        )
                        return None # Indicate that the value was not found and not resolved
            except TypeError as e:
                logger.error(f"[{self.node_name}] TypeError encountered with key '{key}'. Is it hashable? Error: {e}", exc_info=True)
                raise ValueError(f"Invalid key type for lookup: {type(key)}. Key must be hashable.") from e

        elif cache_action == 'store':
            if not isinstance(data, dict) or 'key' not in data or 'value' not in data:
                logger.error(f"[{self.node_name}] Invalid data for 'store' action. Expected dict with 'key' and 'value'. Got: {data}")
                raise ValueError("For 'store' action, 'data' must be a dict containing 'key' and 'value'.")
            
            key_to_store = data['key']
            value_to_store = data['value']

            if not isinstance(key_to_store, (str, int, float, bool, tuple, frozenset, type(None))):
                logger.debug(f"[{self.node_name}] Attempting to store with potentially unhashable key type: {type(key_to_store)}. Proceeding with caution.")
            
            try:
                cache_store[key_to_store] = value_to_store
                logger.info(f"[{self.node_name}] Stored value for key: '{key_to_store}'")
                return value_to_store # Return the stored value
            except TypeError as e:
                logger.error(f"[{self.node_name}] TypeError encountered when storing key '{key_to_store}'. Is it hashable? Error: {e}", exc_info=True)
                raise ValueError(f"Invalid key type for storage: {type(key_to_store)}. Key must be hashable.") from e

        else:
            logger.error(f"[{self.node_name}] Invalid 'cache_action' specified in context: '{cache_action}'. Expected 'lookup' or 'store'.")
            raise ValueError(f"Invalid 'cache_action': '{cache_action}'.")