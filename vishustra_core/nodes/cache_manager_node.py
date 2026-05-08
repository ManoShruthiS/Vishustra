import logging
from typing import Any, Dict, Optional
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A node responsible for managing intermediate state and caching results 
    within the Vishustra orchestration pipeline. It supports retrieval, 
    storage, and invalidation of data based on context-provided keys.
    """

    def __init__(self, default_ttl: Optional[int] = None):
        self._internal_cache: Dict[str, Any] = {}
        self._default_ttl = default_ttl
        logger.debug("CacheManagerNode initialized with internal storage.")

    @property
    def node_name(self) -> str:
        """Returns the canonical name of the node."""
        return "CacheManagerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Coordinates cache operations based on the provided context.
        
        Expected context keys:
            - cache_op: The operation to perform ('get', 'set', 'delete', 'clear').
            - cache_key: The identifier for the cached object.
        
        :param data: The payload to be cached (if operation is 'set').
        :param context: Metadata containing cache configuration and operation instructions.
        :return: The cached value, the original data, or a boolean status depending on the operation.
        """
        operation = context.get("cache_op", "get").lower()
        key = context.get("cache_key")

        if not key and operation in ("get", "set", "delete"):
            logger.error(f"[{self.node_name}] Missing required 'cache_key' in context for operation: {operation}")
            raise KeyError("cache_key is required for CacheManagerNode operations.")

        try:
            if operation == "get":
                return self._handle_get(key)
            elif operation == "set":
                return self._handle_set(key, data)
            elif operation == "delete":
                return self._handle_delete(key)
            elif operation == "clear":
                return self._handle_clear()
            else:
                logger.warning(f"[{self.node_name}] Unsupported operation: {operation}. Passing through data.")
                return data
        except Exception as e:
            logger.exception(f"[{self.node_name}] Error during cache operation '{operation}': {str(e)}")
            raise

    def _handle_get(self, key: str) -> Any:
        """Retrieves a value from the cache."""
        if key in self._internal_cache:
            logger.info(f"[{self.node_name}] Cache HIT for key: {key}")
            return self._internal_cache[key]
        
        logger.info(f"[{self.node_name}] Cache MISS for key: {key}")
        return None

    def _handle_set(self, key: str, value: Any) -> Any:
        """Stores a value in the cache."""
        self._internal_cache[key] = value
        logger.info(f"[{self.node_name}] Successfully cached data under key: {key}")
        return value

    def _handle_delete(self, key: str) -> bool:
        """Removes a specific key from the cache."""
        if key in self._internal_cache:
            del self._internal_cache[key]
            logger.info(f"[{self.node_name}] Evicted key: {key}")
            return True
        logger.debug(f"[{self.node_name}] Attempted to delete non-existent key: {key}")
        return False

    def _handle_clear(self) -> bool:
        """Purges all entries from the cache."""
        self._internal_cache.clear()
        logger.info(f"[{self.node_name}] Cache cleared successfully.")
        return True