import logging
from typing import Any, Dict, Union
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A Vishustra processing node that manages an internal, in-memory cache.

    It supports operations to 'get', 'set', 'delete', and 'clear' cache entries
    based on the 'action' specified in the `context` dictionary.

    Cache keys are expected to be hashable (e.g., str, int, float, tuple).

    Expected `context` parameters and `data` input for `process` method:

    - To 'get' a value:
        `context = {'action': 'get'}`
        `data`: The key (e.g., str, int) to retrieve.
        Returns: The cached value, or `None` if the key is not found or invalid.

    - To 'set' a value:
        `context = {'action': 'set'}`
        `data`: A dictionary like `{'key': 'my_key', 'value': 'my_value'}`.
                The 'key' must be hashable.
        Returns: The value that was successfully set, or `None` on error.

    - To 'delete' a value:
        `context = {'action': 'delete'}`
        `data`: The key to delete from the cache.
        Returns: `None`.

    - To 'clear' the entire cache:
        `context = {'action': 'clear'}`
        `data`: Ignored.
        Returns: `None`.
    """

    def __init__(self):
        super().__init__()
        self._cache: Dict[Any, Any] = {}
        logger.debug(f"CacheManagerNode '{self.node_name}' initialized with an empty cache.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes a cache operation based on the 'action' specified in the context.

        Args:
            data: The input data, which typically serves as the key for 'get'/'delete',
                  or a dictionary containing 'key' and 'value' for 'set'.
            context: A dictionary containing operational parameters.
                     Expected to have a 'action' key with values like 'get', 'set', 'delete', 'clear'.

        Returns:
            The result of the cache operation (e.g., retrieved value, or None).
        """
        action = context.get('action', 'get').lower()
        
        if action == 'get':
            if not isinstance(data, (str, int, float, tuple)): # Valid hashable types
                logger.error(
                    f"CacheManagerNode: 'get' action received invalid key type: {type(data)}. "
                    "Key must be hashable (str, int, float, tuple)."
                )
                return None
            
            key = data
            value = self._cache.get(key)
            if value is not None:
                logger.debug(f"CacheManagerNode: Cache hit for key '{key}'.")
                return value
            else:
                logger.debug(f"CacheManagerNode: Cache miss for key '{key}'.")
                return None

        elif action == 'set':
            if not isinstance(data, dict) or 'key' not in data or 'value' not in data:
                logger.error(
                    f"CacheManagerNode: 'set' action requires 'data' to be a dict "
                    f"with 'key' and 'value' fields. Received: {type(data)}: {data}"
                )
                return None
            
            key = data['key']
            value = data['value']

            if not isinstance(key, (str, int, float, tuple)):
                logger.error(
                    f"CacheManagerNode: 'set' action received invalid key type: {type(key)}. "
                    "Key must be hashable (str, int, float, tuple)."
                )
                return None

            try:
                self._cache[key] = value
                logger.info(f"CacheManagerNode: Set cache for key '{key}'.")
                return value
            except TypeError as e:
                logger.error(f"CacheManagerNode: Failed to set cache for key '{key}' due to type error: {e}")
                return None

        elif action == 'delete':
            if not isinstance(data, (str, int, float, tuple)):
                logger.error(
                    f"CacheManagerNode: 'delete' action received invalid key type: {type(data)}. "
                    "Key must be hashable (str, int, float, tuple)."
                )
                return None

            key = data
            if key in self._cache:
                del self._cache[key]
                logger.info(f"CacheManagerNode: Deleted key '{key}' from cache.")
            else:
                logger.debug(f"CacheManagerNode: Key '{key}' not found for deletion.")
            return None

        elif action == 'clear':
            initial_size = len(self._cache)
            self._cache.clear()
            logger.info(f"CacheManagerNode: Cleared cache. {initial_size} items removed.")
            return None

        else:
            logger.warning(
                f"CacheManagerNode: Unknown cache action '{action}' specified in context. "
                "Returning original data as no operation was performed."
            )
            return data

