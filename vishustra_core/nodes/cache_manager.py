import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A processing node that manages data caching operations within Vishustra.

    This node facilitates 'get' and 'set' operations on a shared cache store,
    which is expected to be provided via the processing context. It offers
    a standardized interface for interacting with a simple key-value cache.

    Input `data` for the `process` method should be a dictionary structured as follows:
    - `"operation"` (str): Must be either "get" to retrieve a value or "set" to store a value.
    - `"key"` (Any): The cache key to perform the operation on.
    - `"value"` (Any, required for "set" operation): The value to store in the cache.

    The `context` dictionary passed to `process` must contain:
    - `"cache_store"`: A mutable dictionary-like object that serves as the cache.

    Returns:
    - For a "get" operation: The cached value if the key is found; otherwise, `None` for a cache miss.
    - For a "set" operation: The value that was successfully stored in the cache.
    """

    def __init__(self):
        """
        Initializes the CacheManagerNode.
        """
        logger.debug("CacheManagerNode initialized.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "CacheManagerNode"

    def process(self, data: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """
        Executes the specified cache operation based on the input data.

        Args:
            data: A dictionary containing the details of the cache operation.
                  Expected keys: "operation", "key", and optionally "value".
            context: A dictionary holding shared runtime resources, crucially
                     including the "cache_store" dictionary.

        Returns:
            Any: The retrieved cache value (or None) for "get", or the stored
                 value for "set" operations.

        Raises:
            ValueError: If the input `data` dictionary is malformed, missing
                        required keys, or specifies an unrecognized operation.
            KeyError: If the "cache_store" is absent from the `context`.
            TypeError: If the object provided as "cache_store" in `context`
                       is not a standard dictionary.
            Exception: Catches any other unforeseen errors during the cache
                       operation to ensure robustness.
        """
        try:
            if not isinstance(data, dict):
                raise ValueError("Input 'data' must be a dictionary.")

            operation = data.get("operation")
            key = data.get("key")

            if operation not in ["get", "set"]:
                raise ValueError(
                    f"Invalid cache operation: '{operation}'. Must be 'get' or 'set'."
                )
            if key is None:
                raise ValueError("Cache operation 'key' cannot be None.")

            if "cache_store" not in context:
                raise KeyError("Missing 'cache_store' in context for CacheManagerNode.")
            
            cache_store = context["cache_store"]
            if not isinstance(cache_store, dict):
                raise TypeError("'cache_store' in context must be a dictionary.")

            if operation == "get":
                if key in cache_store:
                    value = cache_store[key]
                    logger.debug(f"Cache hit for key '{key}'.")
                    return value
                else:
                    logger.debug(f"Cache miss for key '{key}'.")
                    return None
            elif operation == "set":
                if "value" not in data:
                    raise ValueError("Missing 'value' in data for 'set' operation.")
                
                value_to_set = data["value"]
                cache_store[key] = value_to_set
                logger.debug(f"Cached value for key '{key}'.")
                return value_to_set

        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"CacheManagerNode encountered a configuration or input error: {e}")
            raise
        except Exception as e:
            logger.exception(f"An unexpected error occurred in CacheManagerNode: {e}")
            raise