import logging
from typing import Any, Dict, Optional

# Assuming vishustra_core.nodes.base_node exists in the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A Vishustra node that provides in-memory cache management operations.

    This node supports setting, getting, deleting, and clearing cache entries
    based on the 'cache_action' specified in the context.

    Expected context parameters:
    - "cache_action" (str): The desired cache operation ('get', 'set', 'delete', 'clear').
    - "cache_key" (str): The key for the cache entry (required for 'get', 'set', 'delete').
    - "default_value" (Any, optional): A value to return if 'get' action
                                       finds no entry for 'cache_key'.

    The 'data' input to the process method is used as the value for the 'set' action.
    """

    _cache: Dict[str, Any]

    def __init__(self):
        """
        Initializes the CacheManagerNode with an empty in-memory cache.
        """
        self._cache = {}
        logger.debug(f"Initialized {self.node_name}.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "CacheManagerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes the specified cache operation.

        Args:
            data (Any): The input data. Used as the value to set when 'cache_action' is 'set'.
                        Otherwise, it's generally ignored.
            context (Dict[str, Any]): A dictionary containing operational parameters,
                                       including 'cache_action' and 'cache_key'.

        Returns:
            Any: The result of the cache operation:
                 - For 'set': The data that was just set.
                 - For 'get': The retrieved value, or 'default_value' if not found.
                 - For 'delete': True if the key was deleted, False if not found.
                 - For 'clear': True indicating the cache was cleared.

        Raises:
            ValueError: If required context parameters like 'cache_action' or 'cache_key'
                        are missing or invalid.
            Exception: For any unexpected errors during cache operations.
        """
        action: Optional[str] = context.get("cache_action")
        key: Optional[str] = context.get("cache_key")

        if not action:
            logger.error("Missing 'cache_action' in context for CacheManagerNode.")
            raise ValueError("Context parameter 'cache_action' is required.")

        valid_actions = {"get", "set", "delete", "clear"}
        if action not in valid_actions:
            logger.error(f"Invalid 'cache_action' specified: '{action}'. Must be one of {list(valid_actions)}.")
            raise ValueError(f"Invalid cache action: '{action}'.")

        if action in {"get", "set", "delete"} and not key:
            logger.error(f"Missing 'cache_key' in context for CacheManagerNode action '{action}'.")
            raise ValueError(f"Context parameter 'cache_key' is required for action '{action}'.")

        result: Any = None

        try:
            if action == "set":
                self._cache[key] = data
                logger.info(f"Cache: Set key '{key}' with value of type {type(data).__name__}.")
                result = data  # Return the data that was just set
            elif action == "get":
                default_value: Any = context.get("default_value")
                result = self._cache.get(key, default_value)
                if result is default_value:
                    logger.debug(f"Cache: Key '{key}' not found. Returning default value of type {type(default_value).__name__}.")
                else:
                    logger.info(f"Cache: Retrieved key '{key}'. Value type: {type(result).__name__}.")
            elif action == "delete":
                if key in self._cache:
                    del self._cache[key]
                    logger.info(f"Cache: Deleted key '{key}'.")
                    result = True  # Indicate successful deletion
                else:
                    logger.debug(f"Cache: Attempted to delete non-existent key '{key}'.")
                    result = False # Indicate key was not found
            elif action == "clear":
                self._cache.clear()
                logger.info("Cache: All items cleared.")
                result = True # Indicate successful clear
        except Exception as e:
            logger.exception(f"An unexpected error occurred during cache operation '{action}' for key '{key}'.")
            raise

        return result