import logging
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A Vishustra node designed for managing data caching operations within a pipeline.

    This node supports two primary operations, controlled by the 'cache_operation'
    key in the execution context:

    1.  'get': Attempts to retrieve data from the cache using a specified 'cache_key'.
        If the data is found, it is returned. If not, the input `data` (acting as a
        default or fallback value) is returned instead. This allows subsequent nodes
        to either receive cached data or the original data for processing (e.g.,
        computation, then potentially storing it back).

    2.  'set': Stores the input `data` into the cache using the provided 'cache_key'.
        The stored `data` is then returned, allowing the pipeline to continue with
        the value that was just cached.

    The actual cache store is expected to be maintained within the `context` dictionary,
    typically under the key 'global_cache'. This design enables the cache to be shared
    and persist across different node executions within a Vishustra orchestration.

    Configuration expected in the `context` for the `process` method:
    -   'cache_operation' (str): Required. Must be either "get" or "set" (case-insensitive).
    -   'cache_key' (str): Required. A unique identifier used for cache access.
    -   'global_cache' (Dict[str, Any]): Optional. The dictionary serving as the in-memory
                                        cache store. If not present, an empty dictionary
                                        will be initialized in the context for the current
                                        pipeline run, though typically an orchestrator
                                        would pre-populate this.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes the specified cache management operation ('get' or 'set').

        Args:
            data (Any): The input data. Its role depends on the `cache_operation`:
                        - For 'get' operations: This data is returned if a cache miss occurs.
                        - For 'set' operations: This data is the value to be stored in the cache.
            context (Dict[str, Any]): The execution context, which must contain
                                     'cache_operation' and 'cache_key'. It also
                                     holds the 'global_cache' dictionary.

        Returns:
            Any: The result of the cache operation:
                 - Cached data on a 'get' hit.
                 - The original input `data` on a 'get' miss.
                 - The input `data` (after storing) on a 'set' operation.

        Raises:
            ValueError: If 'cache_operation' or 'cache_key' is missing or contains
                        an unsupported value in the context.
            Exception: Propagates any other unexpected errors encountered during
                       cache processing after logging.
        """
        try:
            cache_operation: str = context.get('cache_operation', '').lower()
            cache_key: Optional[str] = context.get('cache_key')

            if not cache_operation:
                logger.error(f"[{self.node_name}] 'cache_operation' is missing in context.")
                raise ValueError("Missing 'cache_operation' in context. Must be 'get' or 'set'.")
            if not cache_key:
                logger.error(
                    f"[{self.node_name}] 'cache_key' is missing in context for operation '{cache_operation}'.")
                raise ValueError("Missing 'cache_key' in context. A unique key is required.")

            # Ensure the cache store exists in the context. If not, initialize it.
            # This allows the node to function even if the orchestrator hasn't
            # explicitly provided a cache.
            if 'global_cache' not in context:
                context['global_cache'] = {}
                logger.warning(
                    f"[{self.node_name}] Initialized 'global_cache' in context as it was missing. "
                    "Consider pre-populating 'global_cache' at the pipeline orchestration level."
                )

            cache_store: Dict[str, Any] = context['global_cache']

            if cache_operation == "get":
                if cache_key in cache_store:
                    cached_value = cache_store[cache_key]
                    logger.debug(f"[{self.node_name}] Cache hit for key: '{cache_key}'.")
                    return cached_value
                else:
                    logger.info(
                        f"[{self.node_name}] Cache miss for key: '{cache_key}'. Returning fallback data."
                    )
                    return data  # Return original data as fallback on cache miss
            elif cache_operation == "set":
                cache_store[cache_key] = data
                logger.debug(f"[{self.node_name}] Data stored in cache for key: '{cache_key}'.")
                return data
            else:
                logger.error(
                    f"[{self.node_name}] Invalid 'cache_operation': '{cache_operation}' "
                    "provided in context."
                )
                raise ValueError(
                    f"Invalid 'cache_operation': '{cache_operation}'. Expected 'get' or 'set'."
                )
        except Exception as e:
            # Log the error with full traceback before re-raising
            logger.error(
                f"[{self.node_name}] Failed to perform cache operation. "
                f"Operation: {context.get('cache_operation', 'N/A')}, "
                f"Key: {context.get('cache_key', 'N/A')}. Error: {e}",
                exc_info=True
            )
            # Re-raise the exception to allow upstream error handling
            raise
