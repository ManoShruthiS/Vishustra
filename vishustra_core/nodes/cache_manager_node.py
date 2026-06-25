
# Standard library imports
import logging
from typing import Any, Dict, Callable

# Project-specific imports
# The BaseNode class is expected to be located here according to project context.
from vishustra_core.nodes.base_node import BaseNode

# Initialize a logger for this module.
logger = logging.getLogger(__name__)


class CacheManager(BaseNode):
    """
    A Vishustra processing node designed to manage and utilize a simple in-memory cache.

    This node implements a read-through caching strategy within its `process` method.
    It attempts to retrieve a value from its internal cache based on a dynamically
    generated key.

    -   **Cache Hit**: If a cached value is found, the node returns the cached value,
        effectively short-circuiting downstream processing for that specific data.
    -   **Cache Miss**: If no cached value is found, the node returns the original
        input data, allowing subsequent nodes in the orchestration pipeline to
        compute the required result.

    The cache hit/miss status, along with the generated cache key, is recorded
    in the `context` dictionary. This information enables an orchestrator or
    other nodes to store computed results back into this `CacheManager` instance
    when a cache miss previously occurred.

    Beyond its `process` method, this node also exposes methods for explicit storage,
    invalidation, and clearing of cache entries, allowing the orchestration layer
    to manage the cache state dynamically.
    """

    # Internal dictionary to simulate a cache store. In a production-grade system,
    # this would typically be replaced by a more robust, persistent, or distributed
    # cache solution (e.g., Redis, Memcached, a dedicated LRU cache implementation,
    # or an external caching service client).
    _cache: Dict[str, Any]

    # A callable function responsible for generating unique keys for cache entries.
    # This key is derived from the input `data` and current `context`.
    _key_generator: Callable[[Any, Dict[str, Any]], str]

    def __init__(self, key_generator: Callable[[Any, Dict[str, Any]], str]):
        """
        Initializes the CacheManager node.

        Args:
            key_generator: A callable that accepts `data: Any` and `context: Dict[str, Any]`
                           as arguments and returns a unique string representing the cache key
                           for the given input. This function must be deterministic;
                           identical inputs (`data` and relevant `context` parts) should
                           always produce the same key for consistent cache behavior.

        Raises:
            TypeError: If the provided `key_generator` is not a callable function.
        """
        if not callable(key_generator):
            logger.error("Attempted to initialize CacheManager with a non-callable key_generator.")
            raise TypeError("key_generator must be a callable function that returns a string key.")

        self._cache = {}
        self._key_generator = key_generator
        logger.info(f"CacheManager initialized. Using key generator: '{self._key_generator.__name__}'.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by attempting to retrieve a value from the cache.

        This method implements a read-through caching pattern:
        1.  Generates a cache key using the provided `_key_generator` based on `data` and `context`.
        2.  Checks if a value associated with this generated key exists in the internal cache.
        3.  **On Cache Hit**: If found, it returns the cached value directly.
            It also updates `context['cache_hit']` to `True` and `context['cache_key']` with the key.
        4.  **On Cache Miss**: If not found, it returns the original input `data`.
            It updates `context['cache_hit']` to `False` and `context['cache_key']` with the key.

        In case of errors during key generation or cache access, the original `data`
        is returned, an error is logged, and details are added to `context['cache_error']`.

        Args:
            data: The input data payload to be processed. This data, along with
                  the `context`, is passed to the `_key_generator` to create a cache key.
            context: A dictionary containing runtime context information. This will be
                     updated with the following fields:
                     - `cache_hit` (bool): `True` if a cached value was returned, `False` otherwise.
                     - `cache_key` (str | None): The key used for the cache lookup, or `None` on key generation error.
                     - `cache_error` (str, optional): An error message if an exception occurred.

        Returns:
            The cached value if a cache hit occurs, otherwise the original input `data`.
            In case of any error during this process, the original `data` is always
            returned to ensure downstream operations can still proceed, if possible.
        """
        # Initialize or reset context flags for cache operation
        context["cache_hit"] = False
        context["cache_key"] = None
        context.pop("cache_error", None)  # Clear any previous error state

        try:
            cache_key = self._key_generator(data, context)

            if not isinstance(cache_key, str):
                error_msg = (f"Cache key generator '{self._key_generator.__name__}' returned a "
                             f"non-string key (type: {type(cache_key)}). Cache bypass initiated.")
                logger.error(error_msg)
                context["cache_error"] = error_msg
                return data

            context["cache_key"] = cache_key

            if cache_key in self._cache:
                cached_value = self._cache[cache_key]
                context["cache_hit"] = True
                logger.debug(f"Cache HIT for key: '{cache_key}'. Returning cached value.")
                return cached_value
            else:
                logger.debug(f"Cache MISS for key: '{cache_key}'. Passing original data for computation.")
                return data
        except Exception as e:
            # Catch any unexpected errors during key generation or cache lookup
            error_msg = (f"An unexpected error occurred during cache processing for data: "
                         f"{data!r}. Error: {type(e).__name__}: {e}")
            logger.exception(error_msg)  # Log full traceback for better debugging
            context["cache_error"] = error_msg
            # On error, always pass through the original data to avoid breaking the pipeline
            return data

    def store_in_cache(self, key: str, value: Any) -> None:
        """
        Explicitly stores a value into the cache under the given key.

        This method is typically invoked by an orchestrator or another node
        after a computation has successfully produced a result (e.g., following
        a cache miss during the `process` phase).

        Args:
            key: The string key under which to store the value.
            value: The data payload to be stored in the cache.

        Raises:
            TypeError: If the provided `key` is not a string.
        """
        if not isinstance(key, str):
            logger.error(f"Attempted to store in cache with a non-string key (type: {type(key)}).")
            raise TypeError("Cache key must be a string.")

        self._cache[key] = value
        logger.debug(f"Successfully stored value for key: '{key}' in cache.")

    def invalidate_cache(self, key: str) -> bool:
        """
        Removes a specific entry from the cache based on its key.

        Args:
            key: The string key of the cache entry to invalidate.

        Returns:
            True if the key was found and successfully removed from the cache.
            False if the key was not found in the cache (no action taken).

        Raises:
            TypeError: If the provided `key` is not a string.
        """
        if not isinstance(key, str):
            logger.error(f"Attempted to invalidate cache with a non-string key (type: {type(key)}).")
            raise TypeError("Cache key must be a string.")

        if key in self._cache:
            del self._cache[key]
            logger.debug(f"Invalidated cache entry for key: '{key}'.")
            return True
        logger.debug(f"Attempted to invalidate non-existent cache key: '{key}'. No action taken.")
        return False

    def clear_cache(self) -> None:
        """
        Clears all entries from the internal cache.

        This method effectively resets the entire cache, removing all stored data.
        """
        initial_size = len(self._cache)
        self._cache.clear()
        logger.info(f"CacheManager: Cleared all {initial_size} entries from the cache.")

