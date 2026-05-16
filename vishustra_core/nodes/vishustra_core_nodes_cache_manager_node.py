import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A data processing node responsible for managing cached responses.

    Checks if a requested item exists in a shared cache store.
    If a cache hit occurs, it retrieves and returns the cached data. 
    If a cache miss occurs, it returns the original data and updates the context.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "CacheManagerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by attempting to retrieve a cached entry.

        This method first validates the presence and type of 'cache_store' and 'cache_key'
        in the provided `context`. If valid, it attempts to fetch data from the cache.
        Upon a cache hit, the cached data is returned directly. Upon a cache miss,
        the original `data` is returned to allow the pipeline to proceed with
        generating the content, and the `context` is updated with the cache status.

        Args:
            data (Any): The input data or request object. In case of a cache miss,
                        this `data` is passed downstream to subsequent nodes for processing.
            context (Dict[str, Any]): A dictionary containing shared pipeline context.
                                     Expected to contain:
                                     - 'cache_store' (dict-like): The object representing the
                                                                  shared cache (e.g., a dictionary,
                                                                  or an object with `__contains__` and `__getitem__`).
                                     - 'cache_key' (str): The unique key used for cache lookup.

        Returns:
            Any: If a cache hit, the retrieved cached data. If a cache miss or an error
                 occurs during caching operations, the original `data` is returned.

        Side Effects:
            - The `context` dictionary is updated with a 'cache_status' key:
              - 'HIT': Data was successfully retrieved from cache.
              - 'MISS': Data was not found in cache.
              - 'ERROR_NO_CACHE_STORE': 'cache_store' was missing or invalid in context.
              - 'ERROR_NO_CACHE_KEY': 'cache_key' was missing or invalid in context.
              - 'ERROR_CACHE_OPERATION': An unexpected error occurred during cache access.
        """
        logger.debug(f"[{self.node_name}] Starting cache lookup for data processing.")

        cache_store = context.get('cache_store')
        if not hasattr(cache_store, '__getitem__') or not hasattr(cache_store, '__contains__'):
            logger.error(
                f"[{self.node_name}] 'cache_store' not found or not a valid dict-like object "
                f"(missing __getitem__ or __contains__) in context. Cannot perform cache operations."
            )
            context['cache_status'] = 'ERROR_NO_CACHE_STORE'
            return data # Pass data through, as caching is not possible

        cache_key = context.get('cache_key')
        if not isinstance(cache_key, str) or not cache_key:
            logger.error(
                f"[{self.node_name}] 'cache_key' not found or not a valid non-empty string in context. "
                f"Cannot perform cache lookup."
            )
            context['cache_status'] = 'ERROR_NO_CACHE_KEY'
            return data # Pass data through, as caching is not possible

        try:
            if cache_key in cache_store:
                cached_result = cache_store[cache_key]
                context['cache_status'] = 'HIT'
                logger.info(f"[{self.node_name}] Cache HIT for key: '{cache_key}'. Returning cached data.")
                return cached_result
            else:
                context['cache_status'] = 'MISS'
                logger.debug(f"[{self.node_name}] Cache MISS for key: '{cache_key}'.")
                return data
        except Exception as e:
            logger.exception(
                f"[{self.node_name}] An unexpected error occurred during cache operation for key '{cache_key}': {e}"
            )
            context['cache_status'] = 'ERROR_CACHE_OPERATION'
            # On error, pass the original data through to prevent blocking the pipeline.
            return data