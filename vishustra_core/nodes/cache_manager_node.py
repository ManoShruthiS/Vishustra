import logging
import time
from typing import Any, Dict, Optional, Tuple

# Assuming this import path exists in the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A processing node that acts as an in-memory cache manager.

    It supports operations like 'set', 'get', 'delete', and 'clear'
    for managing data within its internal cache, including basic
    Time-To-Live (TTL) functionality for cached items.

    Input 'data' and 'context' define the operation and parameters:
    - For 'set': `data` should be `{"key": Any, "value": Any}`, `context` can contain `{"ttl_seconds": Union[int, float]}`.
    - For 'get': `data` should be the `key` to retrieve.
    - For 'delete': `data` should be the `key` to delete.
    - For 'clear': `data` can be ignored.

    The 'operation' is specified via `context["operation"]`.
    """

    def __init__(self):
        """
        Initializes the CacheManagerNode with an empty in-memory cache.
        The cache stores items as `(value, expiration_timestamp_or_None)`.
        """
        self._cache: Dict[Any, Tuple[Any, Optional[float]]] = {}
        logger.debug("CacheManagerNode initialized.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "CacheManagerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to perform cache operations.

        Args:
            data (Any): The primary data for the operation (e.g., key, or key-value pair).
            context (Dict[str, Any]): A dictionary containing control parameters,
                                       most importantly `{"operation": "set"|"get"|"delete"|"clear"}`.
                                       For 'set', `{"ttl_seconds": Union[int, float]}` can also be provided.

        Returns:
            Any: A dictionary indicating the status and result of the operation.
                 E.g., `{"status": "success", "operation": "set", "key": "my_key"}`
                       `{"status": "hit", "operation": "get", "key": "my_key", "value": "cached_data"}`
                       `{"status": "miss", "operation": "get", "key": "my_key", "value": None}`

        Raises:
            ValueError: If the 'operation' is missing, unknown, or required parameters are missing.
            TypeError: If input parameters are of incorrect types.
        """
        operation_raw = context.get("operation")

        if not operation_raw:
            logger.error("CacheManagerNode received a request without a specified 'operation' in context.")
            raise ValueError("Missing 'operation' in context for CacheManagerNode.")

        if not isinstance(operation_raw, str):
            logger.error(f"CacheManagerNode 'operation' must be a string. Received type: {type(operation_raw).__name__}.")
            raise TypeError(f"Invalid type for 'operation'. Expected string, got {type(operation_raw).__name__}.")

        operation = operation_raw.lower()

        try:
            if operation == "set":
                if not isinstance(data, dict):
                    logger.error(f"CacheManagerNode 'set' operation expects 'data' to be a dictionary. Received type: {type(data).__name__}.")
                    raise TypeError("Invalid data format for 'set' operation. Expected dictionary.")

                key = data.get("key")
                value = data.get("value")
                ttl_seconds = context.get("ttl_seconds")

                if key is None or value is None:
                    logger.error(f"CacheManagerNode 'set' operation requires 'key' and 'value' in data. Received data: {data}")
                    raise ValueError("Missing 'key' or 'value' for 'set' operation.")
                
                expiration_time = None
                if ttl_seconds is not None:
                    if not isinstance(ttl_seconds, (int, float)):
                        logger.warning(f"CacheManagerNode: Invalid 'ttl_seconds' type for key '{key}'. Expected int or float, got {type(ttl_seconds).__name__}. Storing without TTL.")
                    elif ttl_seconds <= 0:
                        logger.warning(f"CacheManagerNode: Non-positive 'ttl_seconds' for key '{key}'. Storing without TTL.")
                    else:
                        expiration_time = time.time() + ttl_seconds

                self._cache[key] = (value, expiration_time)
                logger.debug(f"CacheManagerNode: Key '{key}' set with value (and TTL: {ttl_seconds}s if specified).")
                return {"status": "success", "operation": "set", "key": key}

            elif operation == "get":
                key = data # For 'get', data is expected to be the key
                
                if key not in self._cache:
                    logger.debug(f"CacheManagerNode: Cache miss for key '{key}'.")
                    return {"status": "miss", "operation": "get", "key": key, "value": None}
                
                value, expiration_time = self._cache[key]

                if expiration_time is not None and time.time() > expiration_time:
                    del self._cache[key]
                    logger.info(f"CacheManagerNode: Key '{key}' expired and removed. Cache miss.")
                    return {"status": "miss", "operation": "get", "key": key, "value": None, "reason": "expired"}
                
                logger.debug(f"CacheManagerNode: Cache hit for key '{key}'.")
                return {"status": "hit", "operation": "get", "key": key, "value": value}

            elif operation == "delete":
                key = data # For 'delete', data is expected to be the key

                if key in self._cache:
                    del self._cache[key]
                    logger.debug(f"CacheManagerNode: Key '{key}' deleted from cache.")
                    return {"status": "success", "operation": "delete", "key": key}
                else:
                    logger.info(f"CacheManagerNode: Attempted to delete non-existent key '{key}'.")
                    return {"status": "not_found", "operation": "delete", "key": key}

            elif operation == "clear":
                self._cache.clear()
                logger.info("CacheManagerNode: Cache cleared successfully.")
                return {"status": "success", "operation": "clear"}

            else:
                logger.error(f"CacheManagerNode received an unknown operation: '{operation}'.")
                raise ValueError(f"Unknown operation '{operation}' for CacheManagerNode.")
        except Exception as e:
            logger.error(f"CacheManagerNode encountered an error during operation '{operation}': {e}", exc_info=True)
            raise # Re-raise the exception after logging for upstream handling
