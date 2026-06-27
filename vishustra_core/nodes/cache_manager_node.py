import logging
from typing import Any, Dict

# Assuming BaseNode is available in the Vishustra core nodes module
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class CacheManagerNode(BaseNode):
    """
    A Vishustra node that provides in-memory cache management functionalities.

    It supports 'get', 'set', 'invalidate', and 'clear_all' operations on a
    simple dictionary-based cache.
    """

    def __init__(self):
        """
        Initializes the CacheManagerNode with an empty in-memory cache.
        """
        self._cache: Dict[str, Any] = {}
        logger.info("CacheManagerNode initialized with an in-memory cache.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes cache operations based on the input `data`.

        The `data` dictionary must specify an 'action' and relevant parameters.

        Expected `data` formats:
        - Get: `{'action': 'get', 'key': 'your_key'}`
        - Set: `{'action': 'set', 'key': 'your_key', 'value': 'your_value'}`
        - Invalidate: `{'action': 'invalidate', 'key': 'your_key'}`
        - Clear All: `{'action': 'clear_all'}`

        Args:
            data: A dictionary containing the action and its parameters.
            context: A dictionary providing contextual information for the process.
                     (Not directly used by this node but part of the BaseNode interface).

        Returns:
            - For 'get': The cached value if found, otherwise `None`.
            - For 'set': `True` on successful set.
            - For 'invalidate': `True` if the key was found and invalidated, `False` otherwise.
            - For 'clear_all': `True` on successful cache clearing.

        Raises:
            TypeError: If `data` is not a dictionary.
            ValueError: If 'action' is missing, invalid, or required parameters
                        like 'key' or 'value' are missing for a given action.
        """
        if not isinstance(data, dict):
            logger.error(f"Invalid input data type for CacheManagerNode: expected dict, got {type(data)}")
            raise TypeError("CacheManagerNode expects 'data' to be a dictionary.")

        action = data.get('action')
        key = data.get('key')
        
        if not isinstance(action, str):
            logger.error(f"Invalid or missing 'action' in data for CacheManagerNode: {data}")
            raise ValueError(
                "CacheManagerNode requires a string 'action' ('get', 'set', 'invalidate', 'clear_all') "
                "in input data."
            )

        action = action.lower()

        if action == 'get':
            if key is None:
                logger.error(f"Missing 'key' for 'get' action in data: {data}")
                raise ValueError("CacheManagerNode 'get' action requires a 'key'.")
            
            value = self._cache.get(key)
            if value is not None:
                logger.debug(f"Cache hit for key: '{key}'")
            else:
                logger.debug(f"Cache miss for key: '{key}'")
            return value
        
        elif action == 'set':
            if key is None:
                logger.error(f"Missing 'key' for 'set' action in data: {data}")
                raise ValueError("CacheManagerNode 'set' action requires a 'key'.")
            
            # Check for 'value' key existence, even if its value is None
            if 'value' not in data:
                logger.error(f"Missing 'value' for 'set' action in data: {data}")
                raise ValueError("CacheManagerNode 'set' action requires a 'value'.")
            
            value = data['value'] # Use data['value'] to retrieve actual None if it's there
            self._cache[key] = value
            logger.debug(f"Cache set for key: '{key}' with value: '{value}'")
            return True
            
        elif action == 'invalidate':
            if key is None:
                logger.error(f"Missing 'key' for 'invalidate' action in data: {data}")
                raise ValueError("CacheManagerNode 'invalidate' action requires a 'key'.")
            
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"Cache invalidated for key: '{key}'")
                return True
            else:
                logger.debug(f"Attempted to invalidate non-existent key: '{key}'")
                return False
                
        elif action == 'clear_all':
            original_size = len(self._cache)
            self._cache.clear()
            logger.info(f"Cleared all {original_size} items from the cache.")
            return True

        else:
            logger.error(f"Unsupported cache action: '{action}' in data: {data}")
            raise ValueError(
                f"Unsupported cache action: '{action}'. Expected 'get', 'set', 'invalidate', or 'clear_all'."
            )

