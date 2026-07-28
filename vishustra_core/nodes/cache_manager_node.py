import logging
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A processing node designed to manage an in-memory cache for efficient data retrieval.

    This node supports storing, retrieving, and invalidating data based on operations
    specified within the context dictionary. It is stateful, maintaining its cache
    across calls within the same orchestration run.

    Supported operations via `context['operation']`:
    - 'set': Stores the `data` input under the key specified by `context['key']`.
             Returns the value that was successfully set.
    - 'get': Retrieves the value associated with `context['key']`.
             If the key is not found, it returns `context.get('default')`
             if provided, otherwise `None`.
    - 'invalidate': Removes the entry associated with `context['key']` from the cache.
                    Returns the value that was removed, or `None` if the key was not found.

    Context Parameters:
    - 'operation' (str, required): The cache operation to perform ('set', 'get', 'invalidate').
                                   Case-insensitive.
    - 'key' (str, required): The cache key (identifier) for the operation.
    - 'default' (Any, optional): A default value to return for 'get' operations
                                 if the requested key is not found in the cache.
    """

    def __init__(self):
        """
        Initializes the CacheManagerNode, establishing an empty in-memory dictionary
        to serve as its cache store.
        """
        self._cache: Dict[str, Any] = {}
        logger.debug("CacheManagerNode initialized with an empty in-memory cache.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes the specified cache operation based on the provided context.

        Args:
            data (Any): The payload to be stored if the operation is 'set'.
                        This argument is ignored for 'get' and 'invalidate' operations.
            context (Dict[str, Any]): A dictionary containing control parameters for
                                       the cache operation. It MUST include 'operation'
                                       and 'key'. It MAY include 'default' for 'get' operations.

        Returns:
            Any: The outcome of the cache operation:
                 - For 'set': The value that was just stored.
                 - For 'get': The retrieved cached value, the `default` value if provided,
                              or `None` if the key is not found and no default is set.
                 - For 'invalidate': The value that was removed from the cache, or `None`
                                     if the key was not present.

        Raises:
            ValueError: If critical parameters ('operation', 'key') are missing,
                        invalid, or if an unsupported operation is requested.
        """
        operation_str: Optional[str] = context.get('operation')
        key: Optional[str] = context.get('key')
        default_value: Any = context.get('default')

        if not isinstance(operation_str, str) or not operation_str.strip():
            logger.error("Context error: 'operation' key is missing or not a valid string.")
            raise ValueError("Missing or invalid 'operation' in context. Must be a non-empty string.")
        if not isinstance(key, str) or not key.strip():
            logger.error(f"Context error for operation '{operation_str}': 'key' is missing or not a valid string.")
            raise ValueError("Missing or invalid 'key' in context. Must be a non-empty string.")

        operation = operation_str.strip().lower()
        result: Any = None

        if operation == 'set':
            self._cache[key] = data
            result = data
            logger.debug(f"Cache 'set' operation successful for key='{key}'. Stored value type: {type(data).__name__}.")
        elif operation == 'get':
            if key in self._cache:
                result = self._cache[key]
                logger.debug(f"Cache 'get' operation (HIT) for key='{key}'. Retrieved value type: {type(result).__name__}.")
            else:
                result = default_value
                logger.info(f"Cache 'get' operation (MISS) for key='{key}'. Returning default value (if any).")
        elif operation == 'invalidate':
            if key in self._cache:
                result = self._cache.pop(key)
                logger.debug(f"Cache 'invalidate' operation successful for key='{key}'. Removed value type: {type(result).__name__}.")
            else:
                logger.info(f"Cache 'invalidate' operation (NO-OP) for key='{key}'. Key not found in cache.")
        else:
            logger.error(f"Unsupported cache operation '{operation_str}' encountered.")
            raise ValueError(f"Unsupported cache operation: '{operation_str}'. "
                             "Accepted operations are 'set', 'get', or 'invalidate'.")

        return result