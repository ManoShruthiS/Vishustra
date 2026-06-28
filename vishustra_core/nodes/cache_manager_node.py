import logging
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A Vishustra node responsible for managing an in-memory cache.

    It supports 'get', 'set', and 'evict' operations on cached data.
    The cache is internal to the node instance, allowing for dedicated cache
    management within a specific orchestration flow.

    Input 'data' to the process method should be a dictionary with:
    - "operation" (str): 'get', 'set', or 'evict'.
    - "key" (Any): The key to identify the data in the cache.
    - "value" (Any, optional): The value to store for 'set' operations.

    Returns a dictionary indicating the outcome of the operation, e.g.:
    - For 'get': {"status": "hit"|"miss", "key": ..., "value": ...}
    - For 'set': {"status": "success", "operation": "set", "key": ..., "value": ...}
    - For 'evict': {"status": "success"|"not_found", "operation": "evict", "key": ..., "value": ...}
    - For errors: {"status": "error"|"failure", "message": ...}
    """

    _cache: Dict[Any, Any]

    def __init__(self, initial_cache: Optional[Dict[Any, Any]] = None):
        """
        Initializes the CacheManagerNode with an optional pre-populated cache.

        Args:
            initial_cache (Optional[Dict[Any, Any]]): A dictionary to
                                                       initialize the cache with.
        """
        self._cache = initial_cache if initial_cache is not None else {}
        logger.debug(f"{self.node_name} initialized with {len(self._cache)} initial items.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "CacheManagerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the cache operation based on the input data.

        Args:
            data (Any): A dictionary containing 'operation', 'key', and optionally 'value'.
            context (Dict[str, Any]): A dictionary containing contextual information.
                                     (Not directly used by this node's logic but part of signature).

        Returns:
            Dict[str, Any]: A dictionary detailing the result of the cache operation.

        Raises:
            TypeError: If the input 'data' is not a dictionary.
            ValueError: If required keys like 'operation' or 'key' are missing/invalid.
        """
        if not isinstance(data, dict):
            logger.error(f"{self.node_name}: Input 'data' must be a dictionary. Received: {type(data)}")
            raise TypeError("Input 'data' for CacheManagerNode must be a dictionary.")

        operation = data.get("operation")
        key = data.get("key")
        value = data.get("value")

        if not operation or not isinstance(operation, str):
            logger.error(f"{self.node_name}: Missing or invalid 'operation' in data: {data}")
            raise ValueError("Operation 'data[\"operation\"]' is required and must be a string.")

        # Key is required for all supported operations. Value is only for 'set'.
        if operation in ["get", "set", "evict"] and key is None:
            logger.error(f"{self.node_name}: Missing 'key' for operation '{operation}' in data: {data}")
            raise ValueError(f"Key 'data[\"key\"]' is required for operation '{operation}'.")

        result: Dict[str, Any] = {"status": "failure", "message": "Unknown error."}

        try:
            if operation == "get":
                if key in self._cache:
                    cached_value = self._cache[key]
                    logger.debug(f"{self.node_name}: Cache HIT for key '{key}'.")
                    result = {"status": "hit", "key": key, "value": cached_value}
                else:
                    logger.debug(f"{self.node_name}: Cache MISS for key '{key}'.")
                    result = {"status": "miss", "key": key, "value": None}
            elif operation == "set":
                if value is None:
                    logger.error(f"{self.node_name}: Missing 'value' for 'set' operation for key '{key}'.")
                    raise ValueError("Value 'data[\"value\"]' is required for 'set' operation.")
                self._cache[key] = value
                logger.info(f"{self.node_name}: Set cache for key '{key}'.")
                result = {"status": "success", "operation": "set", "key": key, "value": value}
            elif operation == "evict":
                if key in self._cache:
                    evicted_value = self._cache.pop(key)
                    logger.info(f"{self.node_name}: Evicted key '{key}' from cache.")
                    result = {"status": "success", "operation": "evict", "key": key, "value": evicted_value}
                else:
                    logger.warning(f"{self.node_name}: Attempted to evict non-existent key '{key}'.")
                    result = {"status": "not_found", "operation": "evict", "key": key, "value": None}
            else:
                logger.warning(f"{self.node_name}: Unsupported operation: '{operation}'.")
                result = {"status": "failure", "message": f"Unsupported operation: '{operation}'."}
        except Exception as e:
            logger.error(
                f"{self.node_name}: An unexpected error occurred during operation '{operation}' "
                f"for key '{key}': {e}", exc_info=True
            )
            result = {"status": "error", "message": str(e)}

        return result