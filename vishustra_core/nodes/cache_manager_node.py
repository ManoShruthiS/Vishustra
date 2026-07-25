import logging
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A processing node for managing data caching operations within Vishustra.

    This node facilitates storing and retrieving data from a mutable cache store
    provided in the execution context. It acts as an intermediary to abstract
    cache interactions for other nodes in the orchestration graph.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "CacheManagerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes a cache operation ('get' or 'set') based on the provided data
        and context.

        The `data` parameter typically serves as the cache key for the operation.
        The `context` dictionary must contain the following keys:
        - 'cache_store': A mutable dictionary-like object that serves as the cache.
                         This object will be directly modified for 'set' operations.
        - 'cache_action': A string indicating the desired operation, either "get" or "set".

        If 'cache_action' is "set", the `context` must additionally contain:
        - 'cache_value': The value to be stored in the cache.

        Args:
            data: The input data, usually treated as the key for cache operations.
            context: A dictionary containing the operational context, including
                     the 'cache_store', 'cache_action', and 'cache_value' (if setting).

        Returns:
            - If 'cache_action' is "get": The value associated with the key if found,
              otherwise `None`.
            - If 'cache_action' is "set": The key that was successfully set in the cache.

        Raises:
            ValueError: If critical context parameters ('cache_store', 'cache_action',
                        or 'cache_value' for 'set' operations) are missing, malformed,
                        or specify an unsupported action.
        """
        cache_store: Optional[Dict[Any, Any]] = context.get('cache_store')
        cache_action: Optional[str] = context.get('cache_action')

        if not isinstance(cache_store, dict):
            logger.error(
                "Node '%s': 'cache_store' not found or is not a dictionary in context. "
                "Received type: %s", self.node_name, type(cache_store)
            )
            raise ValueError(
                f"Missing or invalid 'cache_store' in context for '{self.node_name}'. "
                "Expected a dictionary-like object."
            )

        if cache_action not in ["get", "set"]:
            logger.error(
                "Node '%s': 'cache_action' is invalid. Expected 'get' or 'set'. "
                "Received: '%s'", self.node_name, cache_action
            )
            raise ValueError(
                f"Invalid 'cache_action' in context for '{self.node_name}'. "
                "Expected 'get' or 'set'."
            )

        key = data

        if cache_action == "get":
            value = cache_store.get(key)
            if value is not None:
                logger.debug("Node '%s': Cache hit for key '%s'.", self.node_name, key)
            else:
                logger.debug("Node '%s': Cache miss for key '%s'.", self.node_name, key)
            return value
        else:  # cache_action == "set"
            cache_value: Any = context.get('cache_value')
            if 'cache_value' not in context:
                logger.error(
                    "Node '%s': 'set' action requires 'cache_value' to be present in context.",
                    self.node_name
                )
                raise ValueError(
                    f"Missing 'cache_value' in context for '{self.node_name}' 'set' action."
                )

            cache_store[key] = cache_value
            logger.debug("Node '%s': Cache set for key '%s'.", self.node_name, key)
            return key  # Return the key to indicate successful storage
