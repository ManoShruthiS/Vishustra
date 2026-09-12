import logging
from typing import Any, Dict, Optional

# Assuming vishustra_core.nodes.base_node exists and contains BaseNode
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A Vishustra processing node designed to manage data caching operations.

    This node offers capabilities to get, set, and invalidate items within a
    shared cache store, which is expected to be provided via the execution context.

    Supported actions for initialization:
    - "get": Retrieves a value from the cache. The 'data' input to `process`
             should be the cache key. Returns the cached value or None if not found.
    - "set": Stores a key-value pair in the cache. The 'data' input to `process`
             must be a dictionary with 'key' and 'value' fields. Returns True on success.
    - "invalidate": Removes a key from the cache. The 'data' input to `process`
                    should be the cache key. Returns True if the key was removed,
                    False otherwise (e.g., key not present).

    The cache store itself is expected to be a mutable dictionary-like object,
    accessible in the `context` dictionary under the key 'cache_store'.
    """

    def __init__(self, action: str):
        """
        Initializes the CacheManagerNode with a specific caching action.

        Args:
            action (str): The cache operation to perform. Must be one of
                          "get", "set", or "invalidate".

        Raises:
            ValueError: If an unsupported `action` is provided during initialization.
        """
        if action not in ["get", "set", "invalidate"]:
            raise ValueError(f"Invalid cache action: '{action}'. Must be 'get', 'set', or 'invalidate'.")
        self._action = action
        logger.debug(f"CacheManagerNode initialized with action: '{self._action}'")

    @property
    def node_name(self) -> str:
        """Returns the name of the node, including its configured action."""
        return f"CacheManager[{self._action.capitalize()}]"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes the configured cache operation based on the initialized action.

        Args:
            data (Any): The input data for the cache operation. Its expected format
                        depends on the `action` configured for this node instance:
                        - For "get" action: The cache key (Any type).
                        - For "set" action: A dictionary `{"key": Any, "value": Any}`.
                        - For "invalidate" action: The cache key (Any type).
            context (Dict[str, Any]): The execution context, which must contain
                                     'cache_store' as a dictionary-like object.

        Returns:
            Any: The result of the cache operation:
                 - For "get": The cached value (Any) or None if the key is not found.
                 - For "set": True if the value was successfully set in the cache.
                 - For "invalidate": True if the key was removed, False if not found.

        Raises:
            RuntimeError: If 'cache_store' is missing from the context or is not a dict.
            ValueError: If the `data` format is incorrect for the specified action.
        """
        # Ensure 'cache_store' is present and a dictionary-like object in context
        if 'cache_store' not in context or not isinstance(context['cache_store'], dict):
            logger.error("Context is missing 'cache_store' or it's not a dictionary-like object for node '%s'.", self.node_name)
            raise RuntimeError(
                f"CacheManagerNode '{self.node_name}' requires a 'cache_store' (dict-like) in the context."
            )

        cache_store = context['cache_store']
        result: Optional[Any] = None

        if self._action == "get":
            key = data
            if key in cache_store:
                result = cache_store[key]
                logger.debug("Cache HIT for key '%s' in node '%s'.", key, self.node_name)
            else:
                logger.debug("Cache MISS for key '%s' in node '%s'.", key, self.node_name)
            return result
        elif self._action == "set":
            if not isinstance(data, dict) or "key" not in data or "value" not in data:
                logger.error(
                    "Invalid data format for 'set' action in node '%s': %s. Expected {'key': ..., 'value': ...}.",
                    self.node_name, data
                )
                raise ValueError(
                    f"For 'set' action, 'data' must be a dictionary with 'key' and 'value' fields."
                )
            
            key_to_set = data["key"]
            value_to_set = data["value"]
            cache_store[key_to_set] = value_to_set
            logger.info("Cache SET: key '%s' updated in node '%s'.", key_to_set, self.node_name)
            return True
        elif self._action == "invalidate":
            key_to_invalidate = data
            if key_to_invalidate in cache_store:
                del cache_store[key_to_invalidate]
                logger.info("Cache INVALIDATED: key '%s' removed in node '%s'.", key_to_invalidate, self.node_name)
                return True
            else:
                logger.debug("Cache INVALIDATE: key '%s' not found, no action taken in node '%s'.", key_to_invalidate, self.node_name)
                return False
        
        # This branch should ideally be unreachable due to `__init__` validation,
        # but included for defensive programming.
        logger.critical(
            "Unhandled action '%s' reached in process method for node '%s'. This indicates a logic error.",
            self._action, self.node_name
        )
        raise RuntimeError(f"Unknown cache action encountered during processing: {self._action}")