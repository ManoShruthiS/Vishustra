
import logging
from typing import Any, Dict, Optional

# Assuming this import path for BaseNode in the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A Vishustra processing node responsible for managing an in-memory cache.

    This node supports common cache operations like 'get', 'set', 'delete', and 'clear'.
    It's designed to be stateless across invocations regarding its internal cache store,
    making it suitable for shared cache instances or simple request-level caching.

    The 'data' input for the process method should be a dictionary specifying the
    cache action and its parameters.
    """

    def __init__(self):
        """
        Initializes the CacheManagerNode.
        A simple in-memory dictionary is used to simulate the cache storage.
        For production, this would typically be backed by a robust caching system
        like Redis or Memcached, potentially injected via context or configuration.
        """
        self._cache: Dict[str, Any] = {}
        logger.debug("CacheManagerNode initialized with an empty in-memory cache.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the cache command provided in 'data'.

        The 'data' parameter is expected to be a dictionary with at least an 'action' key.
        Supported actions:
        - "get": Retrieves a value from the cache. Requires a 'key'.
                 Returns the cached value or None if not found.
                 Example: {"action": "get", "key": "my_data_key"}
        - "set": Stores or updates a value in the cache. Requires 'key' and 'value'.
                 Returns True on success.
                 Example: {"action": "set", "key": "my_data_key", "value": {"item": "A"}}
        - "delete": Removes a value from the cache. Requires a 'key'.
                    Returns True if deleted, False if key was not found.
                    Example: {"action": "delete", "key": "my_data_key"}
        - "clear": Clears the entire cache. No additional parameters needed.
                   Returns True on success.
                   Example: {"action": "clear"}

        Args:
            data (Any): A dictionary representing the cache command.
            context (Dict[str, Any]): A dictionary containing shared context information.
                                     Not directly used by this basic cache implementation,
                                     but available for future extensions (e.g., shared cache client).

        Returns:
            Any: The result of the cache operation (e.g., cached value, boolean for success).

        Raises:
            ValueError: If the 'data' input is malformed or an unsupported action is requested.
        """
        if not isinstance(data, dict):
            logger.error(f"Invalid input data type for CacheManagerNode. Expected dict, got {type(data)}.")
            raise ValueError("CacheManagerNode expects 'data' to be a dictionary.")

        action: Optional[str] = data.get("action")
        key: Optional[str] = data.get("key")
        value: Any = data.get("value")

        if action is None:
            logger.error("Cache command 'action' is missing in input data: %s", data)
            raise ValueError("Cache command requires an 'action' key.")

        try:
            if action == "get":
                if key is None:
                    logger.warning("Attempted 'get' without a 'key'. Input data: %s", data)
                    raise ValueError("Cache 'get' action requires a 'key'.")
                result = self._cache.get(key)
                if result is None:
                    logger.debug("Cache miss for key: '%s'", key)
                else:
                    logger.debug("Cache hit for key: '%s'", key)
                return result
            elif action == "set":
                if key is None or value is None:
                    logger.warning("Attempted 'set' without 'key' or 'value'. Input data: %s", data)
                    raise ValueError("Cache 'set' action requires 'key' and 'value'.")
                self._cache[key] = value
                logger.info("Key '%s' set in cache.", key)
                return True
            elif action == "delete":
                if key is None:
                    logger.warning("Attempted 'delete' without a 'key'. Input data: %s", data)
                    raise ValueError("Cache 'delete' action requires a 'key'.")
                if key in self._cache:
                    del self._cache[key]
                    logger.info("Key '%s' deleted from cache.", key)
                    return True
                else:
                    logger.debug("Attempted to delete non-existent key '%s' from cache.", key)
                    return False
            elif action == "clear":
                self._cache.clear()
                logger.info("Cache has been cleared.")
                return True
            else:
                logger.error("Unsupported cache action '%s' requested.", action)
                raise ValueError(f"Unsupported cache action: '{action}'")
        except Exception as e:
            logger.exception("An error occurred during cache operation for data: %s", data)
            raise RuntimeError(f"Failed to perform cache operation '{action}': {e}") from e

