import logging
from typing import Any, Dict

# Assuming BaseNode is available at this path as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManager(BaseNode):
    """
    A processing node designed to manage a simple in-memory cache within the
    orchestration context. It supports 'get', 'set', 'get_or_compute', and 'clear'
    operations for dynamic data management across nodes.

    The cache store is maintained as a dictionary within the `context` object,
    identified by `_cache_key_in_context`. This allows the cache to persist
    and be accessed by other nodes or subsequent calls within the same run.

    Input `data` dictionary structure for the `process` method:
    - 'operation': (str, required) The desired cache action:
        - 'get': Retrieve a value by 'key'.
        - 'set': Store a 'value' associated with a 'key'.
        - 'get_or_compute': Attempt to retrieve by 'key'. If found, return it.
                            If not found, return the input 'value' for downstream
                            computation.
        - 'clear': Clear a specific 'key' or the entire cache if 'key' is None.
    - 'key': (str, required for 'get', 'set', 'get_or_compute', optional for 'clear')
             The identifier for the cached item.
    - 'value': (Any, required for 'set', optional for 'get_or_compute')
             For 'set', this is the data to be stored.
             For 'get_or_compute', this is the data to be passed to a downstream
             computation node if a cache miss occurs.

    Output of `process` method:
    - For 'get': The cached value if found, otherwise `None`.
    - For 'set': The value that was successfully stored.
    - For 'get_or_compute': The cached value if a hit. If a miss, the original
                            'value' from the input `data` is returned, signaling
                            to the orchestrator that computation is required.
    - For 'clear': `None`.

    Error Handling:
    - Raises `TypeError` if input `data` is not a dictionary.
    - Raises `ValueError` for missing or unsupported 'operation' or 'key'.
    """

    _cache_key_in_context: str = "vishustra_context_cache_store"

    def __init__(self):
        """
        Initializes the CacheManager node.
        No specific constructor parameters for the base implementation,
        but can be extended for advanced cache configurations (e.g., TTL, external backend).
        """
        pass

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "CacheManager"

    def process(self, data: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """
        Executes the specified cache operation within the node's context.

        Args:
            data: A dictionary containing 'operation', 'key', and optionally 'value'.
            context: The shared context dictionary for the current orchestration run.

        Returns:
            The result of the cache operation (e.g., cached value, value to compute).

        Raises:
            TypeError: If `data` is not a dictionary.
            ValueError: If required keys ('operation', 'key') are missing or
                        'operation' is unsupported.
            Exception: Propagates any underlying errors during cache access.
        """
        if not isinstance(data, dict):
            logger.error(
                "Invalid input data type for CacheManager. Expected dict, got %s.",
                type(data).__name__
            )
            raise TypeError("CacheManager expects input data to be a dictionary.")

        operation = data.get("operation")
        key = data.get("key")
        value_for_set_or_compute = data.get("value")

        if not operation:
            logger.error("CacheManager 'operation' key is missing in input data: %s", data)
            raise ValueError("Missing 'operation' in input data.")

        if operation not in ["get", "set", "get_or_compute", "clear"]:
            logger.error("Unsupported cache operation '%s' received.", operation)
            raise ValueError(f"Unsupported cache operation: '{operation}'.")

        if operation != "clear" and not isinstance(key, str):
            logger.error("CacheManager 'key' must be a string for operation '%s', got %s.",
                         operation, type(key).__name__)
            raise ValueError(f"Missing or invalid 'key' for operation '{operation}'. Key must be a string.")
        
        # Ensure the cache store exists in the context
        if self._cache_key_in_context not in context:
            context[self._cache_key_in_context] = {}
            logger.debug("Initialized context cache store at key '%s'.", self._cache_key_in_context)

        cache_store: Dict[str, Any] = context[self._cache_key_in_context]

        try:
            if operation == "get":
                if key in cache_store:
                    logger.debug("Cache hit for key '%s'.", key)
                    return cache_store[key]
                else:
                    logger.debug("Cache miss for key '%s'.", key)
                    return None

            elif operation == "set":
                if "value" not in data:
                    logger.error("CacheManager 'value' key is missing for 'set' operation.")
                    raise ValueError("Missing 'value' for 'set' operation.")
                
                cache_store[key] = value_for_set_or_compute
                logger.debug("Value successfully set in cache for key '%s'.", key)
                return value_for_set_or_compute # Return the stored value as confirmation

            elif operation == "get_or_compute":
                if key in cache_store:
                    logger.debug("Cache hit for key '%s' (get_or_compute).", key)
                    return cache_store[key]
                else:
                    logger.debug("Cache miss for key '%s' (get_or_compute). Returning input 'value' for downstream computation.", key)
                    # On cache miss, return the original 'value' from input data.
                    # This 'value' is expected to be the payload for a downstream
                    # computation node.
                    return value_for_set_or_compute

            elif operation == "clear":
                if key: # Clear a specific key
                    if key in cache_store:
                        del cache_store[key]
                        logger.debug("Cache key '%s' cleared from store.", key)
                    else:
                        logger.debug("Attempted to clear non-existent cache key '%s'.", key)
                else: # Clear the entire cache
                    context[self._cache_key_in_context] = {}
                    logger.debug("All cache cleared from context key '%s'.", self._cache_key_in_context)
                return None # Clear operations typically do not return data

        except Exception as e:
            logger.exception(
                "An unexpected error occurred during CacheManager processing for operation '%s' and key '%s'.",
                operation, key
            )
            raise # Re-raise the exception after logging for upstream handling
