import logging
from typing import Any, Dict, Optional

# Assuming this path exists in the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A processing node designed to manage data caching operations within the Vishustra framework.

    This node facilitates intelligent retrieval, storage, and invalidation
    of data using a shared cache store provided via the execution context.
    It supports 'get', 'set', and 'delete' operations, allowing for flexible
    integration into various orchestration workflows.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "CacheManagerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes a cache operation based on instructions provided in the context.

        The `context` dictionary is expected to contain the following keys for
        cache-related operations:
        - 'cache_store': A mutable dictionary-like object (e.g., dict, LRU cache instance)
                         that serves as the actual cache. This is typically managed
                         by the orchestration layer or a higher-level component.
        - 'cache_key': A string or hashable object representing the unique key
                       for the data within the `cache_store`.
        - 'cache_action': A string indicating the desired cache operation:
                          'get', 'set', 'delete', or 'noop' (default if not specified).

        Args:
            data (Any): The input data for the node.
                        - For 'set' action: This is the value to be stored in the cache.
                        - For 'get' action: This is the data to be returned in case of a
                                           cache miss, allowing the pipeline to continue
                                           with the original input.
                        - For 'delete'/'noop': This data is typically passed through.
            context (Dict[str, Any]): A dictionary containing runtime information,
                                      including the 'cache_store', 'cache_key', and
                                      'cache_action' for this specific operation.

        Returns:
            Any: The result of the cache operation:
                 - For 'get': Returns the cached data if found; otherwise, returns the
                   original input `data` (indicating a cache miss).
                 - For 'set': Returns the original input `data` (which was stored).
                 - For 'delete' or 'noop': Returns the original input `data`.
                 - In case of critical errors or missing configurations, it attempts
                   to return the original `data` to prevent pipeline interruption.
        """
        cache_store: Optional[Dict[str, Any]] = context.get("cache_store")
        cache_key: Optional[str] = context.get("cache_key")
        cache_action: str = context.get("cache_action", "noop").lower()

        if cache_store is None:
            logger.warning(
                f"[{self.node_name}] 'cache_store' not found in context. "
                "Skipping all cache operations and passing data through."
            )
            return data

        # For 'get', 'set', 'delete' actions, a cache_key is mandatory.
        if cache_key is None and cache_action not in ["noop"]:
            logger.warning(
                f"[{self.node_name}] 'cache_key' not found in context for action '{cache_action}'. "
                "Skipping specific cache operation and passing data through."
            )
            return data

        try:
            if cache_action == "get":
                if cache_key is not None and cache_key in cache_store:
                    cached_value = cache_store[cache_key]
                    logger.debug(f"[{self.node_name}] Cache HIT for key '{cache_key}'.")
                    return cached_value
                else:
                    logger.debug(f"[{self.node_name}] Cache MISS for key '{cache_key}'. Returning original data.")
                    return data

            elif cache_action == "set":
                if cache_key is not None:
                    cache_store[cache_key] = data
                    logger.info(f"[{self.node_name}] Data set in cache for key '{cache_key}'.")
                return data

            elif cache_action == "delete":
                if cache_key is not None and cache_key in cache_store:
                    del cache_store[cache_key]
                    logger.info(f"[{self.node_name}] Data deleted from cache for key '{cache_key}'.")
                elif cache_key is not None:
                    logger.debug(f"[{self.node_name}] Attempted to delete non-existent key '{cache_key}'.")
                return data

            elif cache_action == "noop":
                logger.debug(f"[{self.node_name}] Explicit 'noop' cache action. Passing data through.")
                return data

            else:
                logger.warning(
                    f"[{self.node_name}] Unknown cache action '{cache_action}'. "
                    "Passing data through without cache operation."
                )
                return data

        except Exception as e:
            # Catching broad exceptions to ensure pipeline resilience in case of cache issues.
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during cache operation "
                f"'{cache_action}' for key '{cache_key}': {e}", exc_info=True
            )
            # On error, always pass the original data through to avoid breaking the pipeline
            return data