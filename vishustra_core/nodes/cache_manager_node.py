import logging
from typing import Any, Dict, Optional
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    CacheManagerNode handles the persistence and retrieval of intermediate pipeline results.
    It allows the system to skip redundant computations by storing
    and fetching data based on unique identifiers provided in the execution context.
    """

    def __init__(self, storage_backend: Optional[Dict[str, Any]] = None):
        """
        Initializes the CacheManagerNode with an optional storage backend.
        Defaults to an in-memory dictionary if no backend is provided.
        """
        self._storage = storage_backend if storage_backend is not None else {}
        logger.debug("CacheManagerNode initialized with internal storage.")

    @property
    def node_name(self) -> str:
        """
        Returns the unique identifier for this node type.
        """
        return "CacheManagerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by interacting with the cache storage.
        
        The behavior is dictated by context parameters:
        - cache_key (str): The unique key to identify the cached item.
        - cache_action (str): Determines the operation ('get', 'set', 'delete'). 
                             Defaults to 'get' if not specified.

        Returns the cached data if 'get' is successful, otherwise returns the input data.
        """
        cache_key = context.get("cache_key")
        action = context.get("cache_action", "get").lower()

        if not cache_key:
            logger.warning(
                f"[{self.node_name}] Operation '{action}' attempted without a 'cache_key'. "
                "Bypassing cache logic."
            )
            return data

        try:
            if action == "get":
                if cache_key in self._storage:
                    logger.info(f"[{self.node_name}] Cache hit for key: {cache_key}")
                    return self._storage[cache_key]
                
                logger.info(f"[{self.node_name}] Cache miss for key: {cache_key}")
                return data

            elif action == "set":
                self._storage[cache_key] = data
                logger.info(f"[{self.node_name}] Successfully stored data under key: {cache_key}")
                return data

            elif action == "delete":
                removed_val = self._storage.pop(cache_key, None)
                if removed_val is not None:
                    logger.info(f"[{self.node_name}] Evicted key from cache: {cache_key}")
                return data

            else:
                logger.error(f"[{self.node_name}] Invalid cache action requested: {action}")
                return data

        except Exception as e:
            logger.error(
                f"[{self.node_name}] Critical error during cache process for key '{cache_key}': {str(e)}",
                exc_info=True
            )
            # We return data as a fallback to prevent pipeline breakage, 
            # though depending on strictness, we might re-raise.
            return data

    def clear_all(self) -> None:
        """
        Flushes the entire cache storage.
        """
        self._storage.clear()
        logger.info(f"[{self.node_name}] Internal cache storage cleared.")