import logging
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A Vishustra node that manages an in-memory cache.
    It supports 'set', 'get', and 'invalidate' operations to interact
    with a simple key-value store.

    Context Parameters:
    - 'action' (str): Required. Specifies the cache operation ('get', 'set', 'invalidate').
    - 'key' (str): Required. The unique identifier for the cache entry.

    Process Method Behavior:
    - For 'set' action: Caches the `data` provided under the specified 'key' and
      returns the cached `data` itself, allowing it to flow downstream.
    - For 'get' action: Retrieves the value associated with the 'key'.
      Returns the cached value if found, otherwise returns None (cache miss).
    - For 'invalidate' action: Removes the entry associated with the 'key' from the cache.
      Returns None, as this operation primarily affects the cache state, not the data flow.
    """

    def __init__(self):
        super().__init__()
        self._cache: Dict[str, Any] = {}
        logger.info("%s initialized with an empty in-memory cache.", self.node_name)

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes a cache operation ('get', 'set', 'invalidate') based on the context.

        Args:
            data (Any): The primary data input. For 'set' action, this is the value
                        to be stored in the cache. For 'get' or 'invalidate', it's typically
                        ignored, but passed through for 'set'.
            context (Dict[str, Any]): A dictionary providing control parameters:
                - 'action' (str): Must be 'get', 'set', or 'invalidate'.
                - 'key' (str): The unique identifier for the cache entry.

        Returns:
            Any:
                - For 'set': The input `data` that was just cached.
                - For 'get': The value retrieved from the cache, or `None` if not found.
                - For 'invalidate': `None`.

        Raises:
            ValueError: If 'action' or 'key' is missing from the context, or if an
                        unsupported action is specified.
        """
        action = context.get('action')
        key = context.get('key')

        if not action:
            logger.error("%s: 'action' is missing from the context.", self.node_name)
            raise ValueError("Missing 'action' in context for CacheManagerNode operation.")
        if not key:
            logger.error("%s: 'key' is missing from the context for action '%s'.", self.node_name, action)
            raise ValueError("Missing 'key' in context for CacheManagerNode operation.")

        try:
            if action == 'set':
                self._cache[key] = data
                logger.debug("%s: Key '%s' successfully set with data.", self.node_name, key)
                return data
            elif action == 'get':
                if key in self._cache:
                    cached_value = self._cache[key]
                    logger.debug("%s: Cache hit for key '%s'.", self.node_name, key)
                    return cached_value
                else:
                    logger.debug("%s: Cache miss for key '%s'.", self.node_name, key)
                    return None
            elif action == 'invalidate':
                if key in self._cache:
                    del self._cache[key]
                    logger.debug("%s: Key '%s' successfully invalidated.", self.node_name, key)
                else:
                    logger.debug("%s: Attempted to invalidate non-existent key '%s'.", self.node_name, key)
                return None
            else:
                logger.error("%s: Unknown action '%s' received for key '%s'.", self.node_name, action, key)
                raise ValueError(f"Unknown cache action: '{action}'")
        except Exception as e:
            logger.exception("%s: An unexpected error occurred during '%s' operation for key '%s'.",
                             self.node_name, action, key)
            raise # Re-raise the original exception after logging