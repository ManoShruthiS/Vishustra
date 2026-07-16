import logging
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A Vishustra processing node that acts as an in-memory cache manager.
    It supports standard cache operations: 'get', 'set', 'delete', and 'clear'.

    The `process` method expects input `data` as a dictionary, specifying the
    desired cache `action` and associated parameters.

    Input `data` structure examples:
    - {'action': 'set', 'key': <key>, 'value': <value>}
      Stores a value associated with a key in the cache.
    - {'action': 'get', 'key': <key>}
      Retrieves a value by its key from the cache.
    - {'action': 'delete', 'key': <key>}
      Removes a key-value pair from the cache.
    - {'action': 'clear'}
      Empties the entire cache.
    """

    def __init__(self):
        """
        Initializes the CacheManagerNode with an empty, in-memory dictionary
        to serve as the cache store.
        """
        self._cache: Dict[Any, Any] = {}
        logger.debug("CacheManagerNode initialized with an empty in-memory cache.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes the specified cache operation based on the input data.

        Args:
            data: A dictionary containing the cache operation details.
                  Expected keys: 'action' (mandatory), 'key', 'value'.
            context: A dictionary providing additional runtime context.
                     (This node does not directly use the context for cache operations,
                     but it's part of the BaseNode signature.)

        Returns:
            A dictionary detailing the outcome of the cache operation.
            Possible status values: 'success', 'not_found', 'error'.

            Examples of return structures:
            - On 'set' success: {'status': 'success', 'action': 'set', 'key': 'my_key'}
            - On 'get' hit: {'status': 'success', 'action': 'get', 'key': 'my_key', 'value': 'cached_value'}
            - On 'get' miss: {'status': 'not_found', 'action': 'get', 'key': 'non_existent_key'}
            - On 'delete' success: {'status': 'success', 'action': 'delete', 'key': 'my_key'}
            - On 'delete' non-existent: {'status': 'not_found', 'action': 'delete', 'key': 'non_existent_key'}
            - On 'clear' success: {'status': 'success', 'action': 'clear', 'items_removed': 5}
            - On input error: {'status': 'error', 'message': 'Invalid data format or missing required keys'}
            - On unknown action: {'status': 'error', 'message': "Unknown cache action: 'invalid_action'"}
        """
        if not isinstance(data, dict):
            logger.warning("CacheManagerNode received invalid data type: %s. Expected dict.", type(data))
            return {'status': 'error', 'message': 'Input data for CacheManagerNode must be a dictionary.'}

        action: Optional[str] = data.get('action')
        key: Any = data.get('key')
        value: Any = data.get('value')

        if not action:
            logger.warning("Cache operation failed: Missing 'action' key in input data: %s", data)
            return {'status': 'error', 'message': "Missing 'action' key in input data."}

        try:
            if action == 'set':
                if key is None or value is None:
                    logger.warning("Cache 'set' action requires 'key' and 'value'. Received: %s", data)
                    return {'status': 'error', 'action': action, 'message': "Missing 'key' or 'value' for 'set' action."}
                self._cache[key] = value
                logger.info("Cache: Set key='%s'", key)
                return {'status': 'success', 'action': action, 'key': key}

            elif action == 'get':
                if key is None:
                    logger.warning("Cache 'get' action requires 'key'. Received: %s", data)
                    return {'status': 'error', 'action': action, 'message': "Missing 'key' for 'get' action."}
                if key in self._cache:
                    cached_value = self._cache[key]
                    logger.debug("Cache: Hit for key='%s'", key)
                    return {'status': 'success', 'action': action, 'key': key, 'value': cached_value}
                else:
                    logger.debug("Cache: Miss for key='%s'", key)
                    return {'status': 'not_found', 'action': action, 'key': key}

            elif action == 'delete':
                if key is None:
                    logger.warning("Cache 'delete' action requires 'key'. Received: %s", data)
                    return {'status': 'error', 'action': action, 'message': "Missing 'key' for 'delete' action."}
                if key in self._cache:
                    del self._cache[key]
                    logger.info("Cache: Deleted key='%s'", key)
                    return {'status': 'success', 'action': action, 'key': key}
                else:
                    logger.debug("Cache: Attempted to delete non-existent key='%s'", key)
                    return {'status': 'not_found', 'action': action, 'key': key}

            elif action == 'clear':
                initial_size = len(self._cache)
                self._cache.clear()
                logger.info("Cache: Cleared. %d items removed.", initial_size)
                return {'status': 'success', 'action': action, 'items_removed': initial_size}

            else:
                logger.warning("Cache operation failed: Unknown action '%s'. Data: %s", action, data)
                return {'status': 'error', 'message': f"Unknown cache action: '{action}'."}

        except Exception as e:
            logger.exception("An unexpected internal error occurred during cache operation '%s' with data: %s", action, data)
            return {'status': 'error', 'action': action, 'message': f"An internal error occurred: {str(e)}"}