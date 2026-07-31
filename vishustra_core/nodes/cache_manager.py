import logging
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A processing node that provides in-memory key-value caching capabilities.

    This node supports common cache operations such as 'GET', 'SET', 'DELETE',
    and 'CLEAR', allowing other nodes in the orchestration framework to
    efficiently store and retrieve intermediate data.
    """

    def __init__(self):
        """
        Initializes the CacheManagerNode with an empty, in-memory dictionary
        to serve as the cache store.
        """
        self._cache: Dict[str, Any] = {}
        logger.debug(f"[{self.node_name}] Initialized with an empty in-memory cache.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes a specified cache operation based on the 'action' and 'key'
        provided in the context dictionary.

        Args:
            data (Any): The data to be cached if the 'action' is 'SET'.
                        Ignored for 'GET', 'DELETE', and 'CLEAR' actions.
            context (Dict[str, Any]): A dictionary containing parameters for the
                                       cache operation. Must include:
                - 'action' (str): The desired cache operation ('GET', 'SET', 'DELETE', 'CLEAR').
                - 'key' (str, optional): The cache key for 'GET', 'SET', 'DELETE' operations.
                                         Not required for 'CLEAR'.

        Returns:
            Any:
                - For 'GET': The cached value if found, otherwise `None`.
                - For 'SET': The value that was successfully stored in the cache.
                - For 'DELETE': `True` if the key was found and deleted, `False` otherwise.
                - For 'CLEAR': `True` indicating the cache has been cleared.

        Raises:
            ValueError: If 'action' is missing, invalid, or if 'key' is missing
                        for operations that require it.
            Exception: For any unexpected errors during cache operations.
        """
        action = context.get('action')
        key = context.get('key')

        if not isinstance(action, str):
            logger.error(
                f"[{self.node_name}] Validation failed: 'action' must be a string and present in context. Got: {action!r}"
            )
            raise ValueError(
                "Cache operation 'action' must be a string and provided in the context."
            )

        action = action.upper()  # Normalize action string

        # Validate 'key' for actions that require it
        if action in ['GET', 'SET', 'DELETE'] and not isinstance(key, str):
            logger.error(
                f"[{self.node_name}] Validation failed: 'key' must be a string and present in context for action '{action}'. Got: {key!r}"
            )
            raise ValueError(
                f"Cache operation 'key' must be a string and provided in context for action '{action}'."
            )

        try:
            if action == 'GET':
                value = self._cache.get(key)
                if value is not None:
                    logger.debug(f"[{self.node_name}] Cache HIT for key: '{key}'")
                    return value
                else:
                    logger.debug(f"[{self.node_name}] Cache MISS for key: '{key}'")
                    return None
            elif action == 'SET':
                self._cache[key] = data
                logger.info(f"[{self.node_name}] Cache SET: Key '{key}' updated with new data.")
                return data  # Return the value that was set
            elif action == 'DELETE':
                if key in self._cache:
                    del self._cache[key]
                    logger.info(f"[{self.node_name}] Cache DELETE: Key '{key}' removed.")
                    return True
                else:
                    logger.debug(f"[{self.node_name}] Cache DELETE: Key '{key}' not found, no action taken.")
                    return False
            elif action == 'CLEAR':
                self._cache.clear()
                logger.info(f"[{self.node_name}] Cache CLEAR: All entries removed from cache.")
                return True
            else:
                logger.error(
                    f"[{self.node_name}] Invalid cache action specified: '{action}'. "
                    "Supported actions are 'GET', 'SET', 'DELETE', 'CLEAR'."
                )
                raise ValueError(f"Invalid cache action: {action}")
        except Exception as e:
            logger.exception(
                f"[{self.node_name}] An unexpected error occurred during cache operation '{action}' for key '{key}'.",
                exc_info=True
            )
            raise # Re-raise the exception after logging for upstream handling