import logging
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A processing node designed to manage an in-memory cache for Vishustra.
    It supports operations to 'get', 'set', and 'clear' data based on a key.
    The cache implements a basic Least Recently Used (LRU) eviction policy
    if a maximum size is configured.

    Configuration parameters for initialization (passed during node instantiation):
    - `max_size` (Optional[int]): The maximum number of items the cache can hold.
                                  If None or 0, the cache size is unbounded.
                                  Defaults to 100.
    """

    def __init__(self, max_size: Optional[int] = 100):
        """
        Initializes the CacheManagerNode with an optional maximum cache size.

        Args:
            max_size (Optional[int]): The maximum number of items the cache can hold.
                                      If None or 0, the cache size is unbounded.
        """
        # OrderedDict is used to maintain insertion order, allowing for basic LRU eviction.
        self._cache: OrderedDict[Any, Any] = OrderedDict()
        self._max_size = max_size if max_size is not None and max_size > 0 else None
        logger.debug(f"CacheManagerNode initialized with max_size: {self._max_size if self._max_size else 'unbounded'}")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "CacheManager"

    def process(self, key: Any, context: Dict[str, Any]) -> Any:
        """
        Manages cache operations ('get', 'set', 'clear') for a given key.

        The `key` parameter to this method is treated as the cache key.
        The `context` dictionary must contain a 'cache_action' key
        specifying the desired operation.

        Args:
            key (Any): The primary key for the cache operation.
            context (Dict[str, Any]): A dictionary containing cache operation details.
                                       Expected 'cache_action' (str) can be 'get', 'set', or 'clear'.
                                       For 'set' action, the 'value' (Any) to be cached must
                                       also be present in the context.

        Returns:
            Any:
                - For 'get' action: A tuple `(bool_hit, value_or_none)`.
                                   `bool_hit` is True if the key was found, False otherwise.
                                   `value_or_none` is the cached value if found, else None.
                - For 'set' action: The value that was successfully stored in the cache.
                - For 'clear' action: True if the operation was successful.
                                      If clearing a specific key that wasn't found, it still
                                      returns True as the desired state (key not in cache) is achieved.
                                      If `key` is None, the entire cache is cleared.

        Raises:
            ValueError: If 'cache_action' is invalid, or 'value' is missing for a 'set' action.
        """
        cache_action = context.get("cache_action")

        if not isinstance(cache_action, str):
            logger.error(f"Invalid or missing 'cache_action' in context for key '{key}': {context}. "
                         "Expected 'get', 'set', or 'clear'.")
            raise ValueError("Cache operation requires a valid 'cache_action' (string) in context.")

        if cache_action == "get":
            return self._handle_get(key)
        elif cache_action == "set":
            # Check for explicit presence of 'value', as it could legitimately be None.
            if "value" not in context:
                logger.error(f"'value' missing in context for 'set' action with key: '{key}'")
                raise ValueError("Context must contain 'value' for 'set' action.")
            value = context["value"]
            return self._handle_set(key, value)
        elif cache_action == "clear":
            return self._handle_clear(key)
        else:
            logger.error(f"Unknown 'cache_action': '{cache_action}' for key: '{key}'")
            raise ValueError(f"Unknown cache_action: '{cache_action}'. Must be 'get', 'set', or 'clear'.")

    def _handle_get(self, key: Any) -> Tuple[bool, Optional[Any]]:
        """Internal helper to handle the 'get' cache action."""
        if key in self._cache:
            # Move the accessed item to the end to signify recent use (MRU for LRU policy)
            value = self._cache.pop(key)
            self._cache[key] = value
            logger.debug(f"Cache HIT for key: '{key}'")
            return (True, value)
        else:
            logger.debug(f"Cache MISS for key: '{key}'")
            return (False, None)

    def _handle_set(self, key: Any, value: Any) -> Any:
        """Internal helper to handle the 'set' cache action."""
        if key in self._cache:
            # Update existing key and move to end (MRU)
            self._cache.pop(key)
            self._cache[key] = value
            logger.debug(f"Cache UPDATED for key: '{key}'")
        else:
            # Add new key, managing size if bounded
            if self._max_size is not None and len(self._cache) >= self._max_size:
                # Remove the least recently used item (first item in OrderedDict)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                logger.warning(f"Cache overflow: Removed oldest item with key: '{oldest_key}'")
            self._cache[key] = value
            logger.debug(f"Cache SET for key: '{key}'")
        return value

    def _handle_clear(self, key: Any) -> bool:
        """Internal helper to handle the 'clear' cache action."""
        if key is None: # Special case: clear entire cache
            self._cache.clear()
            logger.info("Cache CLEARED entirely.")
            return True
        elif key in self._cache:
            del self._cache[key]
            logger.debug(f"Cache CLEARED for key: '{key}'")
            return True
        else:
            logger.debug(f"Cache CLEAR requested for non-existent key: '{key}'. No action taken.")
            return True # Still return True as the desired state (key not in cache) is achieved.