import logging
from typing import Any, Dict, MutableMapping, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class CacheManager(BaseNode):
    """
    A Vishustra processing node responsible for managing data caching.

    This node implements a 'get-or-set' cache policy:
    - If a 'cache_key' (provided in the context) is found in the cache,
      the cached value is returned directly.
    - If 'cache_key' is not found, the 'data' input to the process method
      is stored in the cache under that key, and then returned.

    The actual cache backend (e.g., an in-memory dictionary, an LRU cache,
    or a proxy to an external cache service) can be injected during
    initialization, defaulting to a simple in-memory dictionary.
    """

    _cache: MutableMapping[str, Any]

    def __init__(self, cache_backend: Optional[MutableMapping[str, Any]] = None):
        """
        Initializes the CacheManager node.

        Args:
            cache_backend (Optional[MutableMapping[str, Any]]): An optional
                mutable mapping object to use as the underlying cache store.
                If `None`, a new standard Python dictionary will be used
                as an in-memory cache. This allows for flexible dependency
                injection of various cache implementations.
        """
        self._cache = cache_backend if cache_backend is not None else {}
        logger.info("CacheManager node initialized. Backend type: %s", type(self._cache).__name__)

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by applying a 'get-or-set' caching strategy.

        This method first attempts to retrieve data from the cache using
        the 'cache_key' provided in the context.
        - On a cache hit, the cached value is returned.
        - On a cache miss, the input `data` is stored in the cache under
          the specified `cache_key` and then returned.

        Args:
            data (Any): The data to be cached if a cache miss occurs.
                        This typically represents the output from a
                        preceding node in the orchestration flow.
            context (Dict[str, Any]): A dictionary containing contextual
                                      information. It *must* include a
                                      'cache_key' (str) to identify
                                      the data within the cache.

        Returns:
            Any: The retrieved value from the cache (on a hit) or the
                 input `data` (on a miss, after being cached).

        Raises:
            ValueError: If 'cache_key' is missing from the context or is not
                        a non-empty string.
            RuntimeError: If an unexpected error occurs during cache operations.
        """
        cache_key = context.get("cache_key")

        if not isinstance(cache_key, str) or not cache_key:
            logger.error("Invalid or missing 'cache_key' in context. Expected a non-empty string.")
            raise ValueError(
                "CacheManager node requires a non-empty string 'cache_key' in the context."
            )

        try:
            # Attempt to retrieve the value from the cache
            if cache_key in self._cache:
                cached_value = self._cache[cache_key]
                logger.debug(f"Cache hit for key: '{cache_key}'. Returning cached value.")
                return cached_value
            else:
                # Cache miss: store the provided data and return it
                self._cache[cache_key] = data
                logger.info(f"Cache miss for key: '{cache_key}'. Storing new data in cache.")
                return data
        except Exception as e:
            logger.exception(
                f"An unexpected error occurred during cache operation for key '{cache_key}': %s", e
            )
            # Re-raise as a RuntimeError to signal a critical failure in the node's operation
            raise RuntimeError(
                f"Failed to perform cache operation for key '{cache_key}': {e}"
            ) from e