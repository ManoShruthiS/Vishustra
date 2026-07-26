import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManager(BaseNode):
    """
    A Vishustra node designed to manage a shared cache store.

    This node provides fundamental caching operations:
    - 'get': Retrieves a value associated with a given key.
    - 'set': Stores a value under a specified key.
    - 'delete': Removes a key-value pair from the cache.
    - 'clear': Empties the entire cache store.
    - 'has': Checks for the existence of a key in the cache.

    The actual cache storage (expected to be a dictionary-like object)
    must be provided in the `context` dictionary under the key "cache_store".
    This allows the orchestration layer to control and share cache instances.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "CacheManager"

    def process(self, data: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """
        Processes cache operations based on the input data and context.

        The `data` input must be a dictionary specifying the `action` and
        any required parameters (e.g., `key`, `value`).

        Supported `data` formats:
        - `{"action": "get", "key": "my_key"}`: Returns the value for "my_key" or `None`.
        - `{"action": "set", "key": "my_key", "value": "my_value"}`: Stores "my_value". Returns `None`.
        - `{"action": "delete", "key": "my_key"}`: Removes "my_key". Returns `None`.
        - `{"action": "clear"}`: Clears the entire cache. Returns `None`.
        - `{"action": "has", "key": "my_key"}`: Returns `True` if "my_key" exists, `False` otherwise.

        The `context` dictionary *must* contain a key named "cache_store"
        whose value is the dictionary-like object serving as the cache.

        Raises:
            ValueError: If `data` is not a dictionary, `action` is invalid,
                        or required parameters like `key`/`value` are missing.
            RuntimeError: If "cache_store" is not found or is not a dictionary in `context`.
        """
        if not isinstance(data, dict):
            logger.error("CacheManager received non-dictionary data: %s", data)
            raise ValueError("Data input for CacheManager must be a dictionary.")

        action = data.get("action")
        key = data.get("key")
        value = data.get("value")

        if "cache_store" not in context or not isinstance(context["cache_store"], dict):
            logger.error("CacheManager failed: 'cache_store' (dict) missing or invalid in context.")
            raise RuntimeError("Cache store not properly initialized in context. Expected 'cache_store' key with a dictionary value.")

        cache: Dict[str, Any] = context["cache_store"]

        logger.debug("CacheManager node executing action: '%s' for key: '%s'", action, key)

        try:
            if action == "get":
                if key is None:
                    logger.warning("CacheManager 'get' action requires a 'key'.")
                    raise ValueError("Missing 'key' parameter for 'get' action.")
                result = cache.get(key)
                logger.debug("Cache 'get' operation for key '%s' returned: %s", key, result)
                return result

            elif action == "set":
                if key is None or value is None:
                    logger.warning("CacheManager 'set' action requires 'key' and 'value'.")
                    raise ValueError("Missing 'key' or 'value' parameter for 'set' action.")
                cache[key] = value
                logger.info("Cache 'set' operation: Key '%s' updated.", key)
                return None

            elif action == "delete":
                if key is None:
                    logger.warning("CacheManager 'delete' action requires a 'key'.")
                    raise ValueError("Missing 'key' parameter for 'delete' action.")
                if key in cache:
                    del cache[key]
                    logger.info("Cache 'delete' operation: Key '%s' removed.", key)
                else:
                    logger.debug("Cache 'delete' operation: Key '%s' not found.", key)
                return None

            elif action == "clear":
                cache.clear()
                logger.info("Cache 'clear' operation: All entries removed.")
                return None

            elif action == "has":
                if key is None:
                    logger.warning("CacheManager 'has' action requires a 'key'.")
                    raise ValueError("Missing 'key' parameter for 'has' action.")
                result = key in cache
                logger.debug("Cache 'has' operation for key '%s' returned: %s", key, result)
                return result

            else:
                logger.error("CacheManager received an unknown action: '%s'", action)
                raise ValueError(f"Unknown cache action: '{action}'. Supported actions are 'get', 'set', 'delete', 'clear', 'has'.")

        except ValueError as e:
            # Re-raise explicit ValueErrors for malformed input
            raise
        except Exception as e:
            # Catch any other unexpected errors during cache operations
            logger.exception("An unexpected error occurred during CacheManager operation for action '%s' on key '%s'.", action, key)
            raise RuntimeError(f"CacheManager operation failed unexpectedly: {e}") from e