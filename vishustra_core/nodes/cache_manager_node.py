import logging
from typing import Any, Dict, Optional

# Assuming BaseNode is available at this path as per project structure.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A processing node designed to manage an in-memory cache.
    It provides operations to store, retrieve, delete, and clear data from its cache.
    This node facilitates efficient data access within an orchestration flow
    by preventing redundant computations or external API calls for frequently
    accessed data.
    """

    def __init__(self, initial_cache: Optional[Dict[Any, Any]] = None):
        """
        Initializes the CacheManagerNode.

        Args:
            initial_cache (Optional[Dict[Any, Any]]): An optional dictionary to use
                                                     as the backing store for the cache.
                                                     If None, an empty dictionary is
                                                     initialized internally. This allows
                                                     for dependency injection of a
                                                     shared cache dictionary if needed.
        """
        self._cache: Dict[Any, Any] = initial_cache if initial_cache is not None else {}
        logger.debug("CacheManagerNode initialized with %d existing items.", len(self._cache))

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "CacheManagerNode"

    def process(self, data: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """
        Executes a specified cache operation based on the input 'data'.

        The 'data' dictionary must contain an 'operation' key.
        Supported operations and their expected 'data' structure:

        - **'get'**: Retrieves a value from the cache.
            `data = {'operation': 'get', 'key': Any}`
            Returns the cached value or `None` if the key is not found.

        - **'set'**: Stores or updates a value in the cache.
            `data = {'operation': 'set', 'key': Any, 'value': Any}`
            Returns the value that was set.

        - **'delete'**: Removes an item from the cache.
            `data = {'operation': 'delete', 'key': Any}`
            Returns `True` if the item was successfully deleted, `False` if the key
            did not exist in the cache.

        - **'clear'**: Clears all items from the cache.
            `data = {'operation': 'clear'}`
            Returns `True`.

        Args:
            data (Dict[str, Any]): A dictionary specifying the cache operation and its
                                   required parameters (e.g., 'key', 'value').
            context (Dict[str, Any]): The current execution context, which may contain
                                      orchestration-specific metadata. This node does
                                      not directly interact with the context for cache storage.

        Returns:
            Any: The result of the cache operation, which varies by operation type.

        Raises:
            ValueError: If the 'data' format is invalid, 'operation' is missing,
                        or an unknown operation is specified.
            KeyError: If a required key for an operation (e.g., 'key' for 'get')
                      is missing from the 'data' dictionary.
        """
        if not isinstance(data, dict):
            logger.error("Invalid input data type for CacheManagerNode. Expected dict, got %s.", type(data))
            raise ValueError(f"Invalid data format: 'data' must be a dictionary. Received: {type(data).__name__}")

        operation = data.get("operation")
        if operation is None:
            logger.error("Missing 'operation' key in input data for CacheManagerNode. Data: %s", data)
            raise ValueError("Missing 'operation' key in input data for CacheManagerNode.")

        logger.debug("CacheManagerNode executing operation: '%s'", operation)

        try:
            if operation == "get":
                key = data.get("key")
                if key is None:
                    logger.error("Missing 'key' for 'get' operation. Input data: %s", data)
                    raise KeyError("Missing 'key' for 'get' operation.")
                
                value = self._cache.get(key)
                if value is None:
                    logger.debug("Cache miss for key: '%s'.", key)
                else:
                    logger.debug("Cache hit for key: '%s'.", key)
                return value

            elif operation == "set":
                key = data.get("key")
                value_to_set = data.get("value")
                if key is None:
                    logger.error("Missing 'key' for 'set' operation. Input data: %s", data)
                    raise KeyError("Missing 'key' for 'set' operation.")
                
                self._cache[key] = value_to_set
                logger.debug("Set cache key '%s' with value. Current cache size: %d", key, len(self._cache))
                return value_to_set

            elif operation == "delete":
                key = data.get("key")
                if key is None:
                    logger.error("Missing 'key' for 'delete' operation. Input data: %s", data)
                    raise KeyError("Missing 'key' for 'delete' operation.")
                
                if key in self._cache:
                    del self._cache[key]
                    logger.debug("Deleted cache key '%s'. Current cache size: %d", key, len(self._cache))
                    return True
                logger.debug("Attempted to delete non-existent cache key: '%s'.", key)
                return False

            elif operation == "clear":
                initial_size = len(self._cache)
                self._cache.clear()
                logger.debug("Cache cleared. %d items removed.", initial_size)
                return True

            else:
                logger.error("Unknown cache operation '%s' specified. Supported: 'get', 'set', 'delete', 'clear'.", operation)
                raise ValueError(f"Unknown cache operation: '{operation}'.")

        except (ValueError, KeyError) as e:
            # Re-raise specific errors after logging for upstream handling
            logger.exception("Validation error during CacheManagerNode process for operation '%s'.", operation)
            raise
        except Exception as e:
            # Catch any unexpected errors during cache operations
            logger.exception("An unexpected error occurred during CacheManagerNode operation '%s' with data %s.", operation, data)
            raise # Re-raise to ensure the orchestration flow is aware of the failure
