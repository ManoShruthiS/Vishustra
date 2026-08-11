import logging
import threading
import time
from typing import Any, Dict, Optional, Callable, Hashable

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A processing node that manages a time-based in-memory cache for data produced
    by an underlying resolver.

    This node intercepts calls, checks if the result for a given key is in the cache
    and is still valid (not expired). If found, it returns the cached value.
    Otherwise, it delegates to a `cache_resolver` callable, stores its result
    in the cache with a Time-To-Live (TTL), and then returns it.

    Parameters:
        cache_resolver (Callable[[Any, Dict[str, Any]], Any]):
            A callable (e.g., another node's process method or a function) that
            will be invoked to produce the value when a cache miss occurs or
            the cached value has expired. It must accept `data` and `context`
            as arguments.
        ttl (int):
            The Time-To-Live in seconds for cache entries. After this duration,
            a cached item is considered stale and will be re-resolved. Defaults to 300 seconds.
        max_entries (Optional[int]):
            Maximum number of entries the cache can hold. If exceeded, the
            oldest entry (based on creation time) will be evicted. If None,
            the cache size is unbounded. Defaults to None.
    """

    def __init__(self,
                 cache_resolver: Callable[[Any, Dict[str, Any]], Any],
                 ttl: int = 300,
                 max_entries: Optional[int] = None):
        if not callable(cache_resolver):
            raise TypeError("cache_resolver must be a callable function or method.")
        if not isinstance(ttl, int) or ttl <= 0:
            raise ValueError("ttl must be a positive integer.")
        if max_entries is not None and (not isinstance(max_entries, int) or max_entries <= 0):
            raise ValueError("max_entries must be a positive integer or None.")

        self._cache_resolver = cache_resolver
        self._ttl = ttl
        self._max_entries = max_entries
        self._cache: Dict[Hashable, Dict[str, Any]] = {}
        self._cache_lock = threading.Lock()
        logger.info(f"CacheManagerNode initialized with TTL: {ttl}s, Max Entries: {max_entries}")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "CacheManager"

    def _generate_key(self, data: Any, context: Dict[str, Any]) -> Hashable:
        """
        Generates a cache key from the input data and context.
        Prioritizes `context['cache_key']` if present. Otherwise, attempts
        to use `data` directly.

        Raises:
            TypeError: If a suitable hashable key cannot be generated.
        """
        # Allow an explicit cache key to be passed in the context
        if 'cache_key' in context:
            key = context['cache_key']
            if not isinstance(key, Hashable):
                raise TypeError(f"Provided 'cache_key' in context is not hashable: {type(key)}")
            return key

        # Otherwise, attempt to use the data itself as the key
        try:
            if isinstance(data, (dict, list, set)):
                # For mutable types, attempt a stable hash.
                # A more sophisticated key generation for complex objects
                # might be needed for real-world scenarios.
                if isinstance(data, dict):
                    return hash(frozenset(sorted(data.items())))
                elif isinstance(data, list):
                    return hash(tuple(data))
                else: # set
                    return hash(frozenset(data))
            else:
                return hash(data)
        except TypeError as e:
            raise TypeError(
                f"Input data is not hashable and no 'cache_key' was provided in context. "
                f"Unable to generate cache key. Consider passing a hashable 'cache_key' in the context. Original error: {e}"
            )

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, either by retrieving from cache or
        by resolving it and then caching the result.

        Args:
            data (Any): The input data for the resolver, often used to derive the cache key.
            context (Dict[str, Any]): The execution context, which may contain
                                       `'cache_key'` to explicitly define the key,
                                       or `'cache_bypass'` to force a re-resolution.

        Returns:
            Any: The cached or newly resolved result.

        Raises:
            Exception: Any exception raised by the underlying `cache_resolver`.
            TypeError: If a cache key cannot be generated.
        """
        cache_key: Hashable
        try:
            cache_key = self._generate_key(data, context)
        except TypeError as e:
            logger.error(f"Failed to generate cache key: {e}")
            raise

        cache_bypass = context.get('cache_bypass', False)

        with self._cache_lock:
            # Check for cache hit
            cached_item = self._cache.get(cache_key)
            current_time = time.time()

            if not cache_bypass and cached_item and (current_time - cached_item['timestamp']) < self._ttl:
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached_item['value']
            
            # Cache miss or expired or bypass requested
            if cache_bypass:
                logger.info(f"Cache bypass requested for key: {cache_key}")
            elif cached_item:
                logger.debug(f"Cache expired for key: {cache_key}. Resolving...")
            else:
                logger.debug(f"Cache miss for key: {cache_key}. Resolving...")

            # If cache size limit is hit, evict the oldest entry
            if self._max_entries is not None and len(self._cache) >= self._max_entries and cache_key not in self._cache:
                oldest_key = min(self._cache, key=lambda k: self._cache[k]['timestamp'])
                logger.debug(f"Cache full, evicting oldest entry: {oldest_key}")
                del self._cache[oldest_key]

            # Resolve the value using the underlying resolver
            try:
                resolved_value = self._cache_resolver(data, context)
            except Exception as e:
                logger.error(f"Error resolving value for key '{cache_key}': {e}")
                raise

            # Store the resolved value in cache
            self._cache[cache_key] = {
                'value': resolved_value,
                'timestamp': current_time
            }
            logger.debug(f"Value resolved and cached for key: {cache_key}")
            return resolved_value
