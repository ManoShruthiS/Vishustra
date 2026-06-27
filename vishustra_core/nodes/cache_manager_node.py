import logging
from typing import Any, Dict, Optional

# Assuming vishustra_core.nodes.base_node is discoverable in the Python path
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A Vishustra processing node that acts as an in-memory cache manager.

    It supports operations such as 'get', 'set', 'delete', and 'clear'
    on its internal cache, driven by commands provided in the `context` dictionary.
    """

    def __init__(self):
        """
        Initializes the CacheManagerNode with an empty in-memory dictionary
        to serve as the cache store.
        """
        self._cache: Dict[str, Any] = {}
        logger.debug("CacheManagerNode initialized with an empty in-memory cache.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes cache commands and manipulates the internal cache.

        The `context` dictionary is expected to contain a 'cache_command' key
        specifying the desired operation ('get', 'set', 'delete', 'clear').
        For 'get', 'set', and 'delete' commands, a 'cache_key' is also required.
        For 'set', the `data` parameter itself is used as the value to be cached.

        Args:
            data: The input data. For 'set' operations, this is the value
                  to be stored in the cache. For other operations, it might
                  represent an upstream result, but its direct use by the
                  cache command is minimal, as operations are primarily
                  driven by `context`.
            context: A dictionary containing operational parameters for the node.
                     Expected keys:
                     - 'cache_command' (str): The command to execute.
                       Valid commands: 'get', 'set', 'delete', 'clear'.
                     - 'cache_key' (str, optional): The key associated with the
                       cache entry for 'get', 'set', and 'delete' commands.

        Returns:
            Any: The result of the cache operation:
                 - For 'get': The cached value if found, otherwise `None`.
                 - For 'set', 'delete', 'clear': `True` upon successful operation,
                   `False` if an operation failed (e.g., key not found for 'delete',
                   or missing required parameters).

        Raises:
            ValueError: If an unknown or unsupported 'cache_command' is provided.
        """
        command: str = context.get("cache_command", "").lower()
        cache_key: Optional[str] = context.get("cache_key")

        if not command:
            logger.warning(
                "CacheManagerNode received a process call without 'cache_command' in context. "
                "Returning original data as a no-op."
            )
            # If no command, it's a no-op from cache perspective, potentially just pass through
            return data

        if command == "get":
            if cache_key is None:
                logger.error("Cache 'get' command received without 'cache_key' in context.")
                return None  # Indicate failure to retrieve due to missing key
            
            value = self._cache.get(cache_key)
            if value is not None:
                logger.info(f"Cache hit for key: '{cache_key}'.")
                return value
            else:
                logger.debug(f"Cache miss for key: '{cache_key}'.")
                return None

        elif command == "set":
            if cache_key is None:
                logger.error("Cache 'set' command received without 'cache_key' in context.")
                return False  # Indicate failure to set
            
            # The 'data' parameter is the value to be cached for 'set' operations.
            self._cache[cache_key] = data
            logger.info(f"Cache set for key: '{cache_key}'.")
            return True

        elif command == "delete":
            if cache_key is None:
                logger.error("Cache 'delete' command received without 'cache_key' in context.")
                return False  # Indicate failure to delete
            
            if cache_key in self._cache:
                del self._cache[cache_key]
                logger.info(f"Cache entry deleted for key: '{cache_key}'.")
                return True
            else:
                logger.debug(f"Attempted to delete non-existent cache key: '{cache_key}'.")
                return False  # Indicate key was not found

        elif command == "clear":
            self._cache.clear()
            logger.info("Cache cleared entirely.")
            return True

        else:
            logger.error(f"Received unknown or unsupported cache command: '{command}'.")
            raise ValueError(f"Unknown cache command received: '{command}'.")