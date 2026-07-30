import logging
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A processing node designed to manage an in-memory cache within the Vishustra
    processing context. This node supports common cache operations such as
    retrieving, storing, and deleting data by a specified key.

    The cache state is maintained within the `context` dictionary under the
    key `'cache_manager_store'`. If this key is not found in the context
    when `process` is called, an empty dictionary will be initialized for it.

    Input 'data' for the `process` method is expected to be a dictionary
    specifying the desired cache operation:
    - **Get Operation**: `{'action': 'get', 'key': str}`
      Attempts to retrieve the value associated with 'key'. Returns the value
      if found, otherwise returns `None` (cache miss).
    - **Set Operation**: `{'action': 'set', 'key': str, 'value': Any}`
      Stores 'value' with the given 'key'. Returns the 'value' that was stored.
      Note: Explicitly storing `None` as a value is supported.
    - **Delete Operation**: `{'action': 'delete', 'key': str}`
      Removes the key-value pair associated with 'key'. Returns the deleted
      value if the key existed, otherwise returns `None`.

    Robust error handling is implemented to validate input data and context
    structures, ensuring reliable operation within an orchestration flow.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "CacheManagerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes cache management operations based on the input data.

        Args:
            data: A dictionary containing the 'action', 'key', and optionally 'value'
                  for the cache operation.
                  Expected formats:
                  - `{'action': 'get', 'key': 'my_key'}`
                  - `{'action': 'set', 'key': 'my_key', 'value': 'my_value'}`
                  - `{'action': 'delete', 'key': 'my_key'}`
            context: The shared processing context dictionary. This dictionary
                     will contain or be initialized with the cache store
                     under the key `'cache_manager_store'`.

        Returns:
            The outcome of the cache operation:
            - For 'get': The cached value or `None` on miss.
            - For 'set': The value that was stored.
            - For 'delete': The value that was deleted or `None` if the key didn't exist.

        Raises:
            TypeError: If the 'context' provided is not a dictionary.
            ValueError: If 'data' is not a dictionary, or if required keys
                        ('action', 'key') are missing, or if an unknown action
                        is specified, or if 'value' is missing for a 'set' action.
            Exception: Propagates any underlying exceptions during cache operations.
        """
        if not isinstance(context, dict):
            logger.error("Vishustra context must be a dictionary.")
            raise TypeError("Context must be a dictionary for CacheManagerNode.")

        # Ensure the cache store is initialized in the context
        if 'cache_manager_store' not in context:
            context['cache_manager_store'] = {}
            logger.debug("Initialized 'cache_manager_store' in context for CacheManagerNode.")

        cache_store: Dict[str, Any] = context['cache_manager_store']

        if not isinstance(data, dict):
            logger.error(f"Invalid input data format for CacheManagerNode. Expected a dictionary, received {type(data)}.")
            raise ValueError(f"CacheManagerNode received invalid data type. Expected dict, got {type(data)}.")

        action: Optional[str] = data.get('action')
        key: Optional[str] = data.get('key')

        if not action or not key:
            logger.error(f"Missing required keys 'action' or 'key' in input data: {data}")
            raise ValueError("CacheManagerNode requires 'action' and 'key' in input data.")

        # Normalize action to lowercase for consistent handling
        action = action.lower()

        try:
            if action == 'get':
                result = cache_store.get(key)
                if result is not None:
                    logger.debug(f"Cache hit for key: '{key}'")
                else:
                    logger.debug(f"Cache miss for key: '{key}'")
                return result
            elif action == 'set':
                # Differentiate between a missing 'value' key and an explicit 'None' value
                if 'value' not in data:
                    logger.error(f"Missing 'value' for 'set' action with key: '{key}' in data: {data}")
                    raise ValueError("CacheManagerNode 'set' action requires a 'value' key in input data.")
                
                value = data['value'] # Retrieve value, can be None explicitly
                cache_store[key] = value
                logger.debug(f"Cache set for key: '{key}'. Value: {value!r}")
                return value
            elif action == 'delete':
                if key in cache_store:
                    result = cache_store.pop(key)
                    logger.debug(f"Cache deleted for key: '{key}'. Deleted value: {result!r}")
                    return result
                else:
                    logger.debug(f"Attempted to delete non-existent key: '{key}'. No action taken.")
                    return None
            else:
                logger.error(f"Unknown cache action: '{action}' received for key: '{key}'. Supported actions are 'get', 'set', 'delete'.")
                raise ValueError(f"Unknown cache action: '{action}'. Supported actions are 'get', 'set', 'delete'.")
        except Exception as e:
            logger.exception(f"An unexpected error occurred during cache operation '{action}' for key '{key}'.")
            raise # Re-raise the exception after logging for upstream handling
