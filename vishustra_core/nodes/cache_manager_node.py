import logging
import time
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class CacheManagerNode(BaseNode):
    """
    A Vishustra processing node responsible for managing data caching operations.

    This node provides functionality to retrieve, store, and invalidate data within
    an in-memory cache, offering a foundational component for optimizing LLM
    orchestration workflows by reducing redundant computations.

    The node's behavior is primarily driven by directives passed within the `context`
    dictionary, allowing for flexible cache interactions based on workflow needs.

    Configuration parameters for initialization:
    - cache_name (str): A unique identifier for this cache instance. Useful for
                        distinguishing between multiple cache managers or policies.
                        Defaults to "default_cache".
    - default_ttl_seconds (Optional[int]): The default time-to-live (TTL) for
                                            cached items, in seconds. If `None`,
                                            cached items will not expire automatically.
    - cache_key_field (Optional[str]): Specifies a field within the input `data`
                                        (if `data` is a dictionary) to use as the
                                        cache key. If `None`, the node will attempt
                                        to derive a key from `context['cache_key']`,
                                        `data['id']`, or the `data` itself.
    - initial_cache_data (Optional[Dict[Any, Any]]): A dictionary to pre-populate
                                                      the cache upon initialization.
    """

    def __init__(self,
                 cache_name: str = "default_cache",
                 default_ttl_seconds: Optional[int] = None,
                 cache_key_field: Optional[str] = None,
                 initial_cache_data: Optional[Dict[Any, Any]] = None):

        self._cache_name = cache_name
        self._default_ttl_seconds = default_ttl_seconds
        self._cache_key_field = cache_key_field

        # Internal in-memory cache for storing values
        self._cache: Dict[Any, Any] = initial_cache_data if initial_cache_data is not None else {}
        # Stores the Unix timestamp (float) of when an item was last added or updated
        self._timestamps: Dict[Any, float] = {}

        # Initialize timestamps for any pre-populated data if TTL is active
        if self._default_ttl_seconds is not None and initial_cache_data is not None:
            current_time = time.time()
            for key in initial_cache_data:
                self._timestamps[key] = current_time

        logger.debug(f"CacheManagerNode '{self.node_name}' initialized. "
                     f"Default TTL: {default_ttl_seconds if default_ttl_seconds is not None else 'None'}s. "
                     f"Key field preference: '{cache_key_field if cache_key_field else 'auto-detect'}'.")

    @property
    def node_name(self) -> str:
        """Returns the programmatic name of the node, including its configured cache name."""
        return f"CacheManagerNode_{self._cache_name}"

    def _get_cache_key(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Determines the appropriate cache key for an operation based on priority:
        1. An explicit 'cache_key' provided in the `context`.
        2. The value associated with `self._cache_key_field` if `data` is a dictionary.
        3. The value of the 'id' field if `data` is a dictionary.
        4. The `data` itself, if it is hashable.

        If no suitable hashable key can be determined, it logs a warning and returns `None`.
        """
        # Priority 1: Explicit 'cache_key' in context
        if 'cache_key' in context:
            return context['cache_key']

        # Priority 2: Configured cache_key_field if data is a dictionary
        if isinstance(data, dict) and self._cache_key_field and self._cache_key_field in data:
            return data[self._cache_key_field]

        # Priority 3: Default 'id' field if data is a dictionary
        if isinstance(data, dict) and 'id' in data:
            return data['id']

        # Priority 4: Use the data itself, if hashable
        try:
            hash(data)  # Check if data is hashable for direct use as a key
            return data
        except TypeError:
            logger.warning(
                f"CacheManagerNode '{self.node_name}': Failed to determine a cache key. "
                f"Input data of type '{type(data).__name__}' is not hashable, "
                f"and no specific key field or 'cache_key' in context was provided. "
                "Caching operations requiring a key will be skipped."
            )
            return None  # Indicate no valid key found

    def _is_expired(self, key: Any) -> bool:
        """
        Checks if a cached item associated with the given key has expired based on its TTL.
        Returns `True` if expired or no timestamp exists for a TTL-configured cache, `False` otherwise.
        """
        if self._default_ttl_seconds is None:
            return False  # No TTL configured, so items do not expire

        timestamp = self._timestamps.get(key)
        if timestamp is None:
            # If no timestamp exists for a key in a TTL-configured cache,
            # consider it expired to ensure it's re-cached with a fresh timestamp.
            return True

        return (time.time() - timestamp) > self._default_ttl_seconds

    def _get_item(self, cache_key: Any) -> Any:
        """
        Retrieves an item from the cache. Performs an expiration check if TTL is active.
        If the item is expired, it's removed from the cache before returning `None`.
        """
        if cache_key is None or cache_key not in self._cache:
            return None

        if self._is_expired(cache_key):
            logger.debug(f"CacheManagerNode '{self.node_name}': Item for key '{cache_key}' expired. Evicting.")
            del self._cache[cache_key]
            if cache_key in self._timestamps:
                del self._timestamps[cache_key]
            return None

        return self._cache[cache_key]

    def _set_item(self, cache_key: Any, value: Any) -> None:
        """
        Stores an item in the cache. Updates its timestamp if TTL is enabled.
        Logs a warning if an attempt is made to set an item with a `None` key.
        """
        if cache_key is None:
            logger.warning(f"CacheManagerNode '{self.node_name}': Attempted to set item with a None key. Skipping operation.")
            return

        self._cache[cache_key] = value
        if self._default_ttl_seconds is not None:
            self._timestamps[cache_key] = time.time()
        logger.debug(f"CacheManagerNode '{self.node_name}': Item for key '{cache_key}' successfully set/updated.")

    def _invalidate_item(self, cache_key: Any) -> None:
        """
        Removes an item from the cache. Logs if the item did not exist.
        Logs a warning if an attempt is made to invalidate an item with a `None` key.
        """
        if cache_key is None:
            logger.warning(f"CacheManagerNode '{self.node_name}': Attempted to invalidate item with a None key. Skipping operation.")
            return

        if cache_key in self._cache:
            del self._cache[cache_key]
            if cache_key in self._timestamps:
                del self._timestamps[cache_key]
            logger.debug(f"CacheManagerNode '{self.node_name}': Item for key '{cache_key}' successfully invalidated.")
        else:
            logger.debug(f"CacheManagerNode '{self.node_name}': Attempted to invalidate non-existent key '{cache_key}'. No action taken.")

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by performing a cache operation defined in the `context`.

        The `context` dictionary can specify the 'cache_operation' (defaults to 'get')
        and an explicit 'cache_key'. The node updates the `context` with a
        'cache_hit' boolean and 'cache_key_used' for observability.

        Supported 'cache_operation' values:
        - 'get' (default): Attempts to retrieve data from the cache. If a hit occurs
                           (and not expired), returns the cached value. On a miss
                           or expiration, returns the original `data`.
        - 'set': Stores the input `data` in the cache using the derived key. Returns
                 the original `data`.
        - 'invalidate': Removes the item associated with the derived key from the cache.
                        Returns the original `data`.
        - 'passthrough': Ignores all cache logic and returns the original `data` directly.

        Args:
            data (Any): The input data to be processed or cached.
            context (Dict[str, Any]): A dictionary for shared state and directives.

        Returns:
            Any: The processed data, which could be the cached value, the original
                 input data, or the result of a cache management action.
        """
        cache_operation = context.get('cache_operation', 'get').lower()
        cache_key = self._get_cache_key(data, context)
        context['cache_key_used'] = cache_key

        # Operations that require a valid cache key
        if cache_key is None and cache_operation not in ['passthrough']:
            logger.error(
                f"CacheManagerNode '{self.node_name}': Cannot perform '{cache_operation}' operation "
                "as a valid cache key could not be determined. Returning original data without caching."
            )
            context['cache_hit'] = False
            return data

        if cache_operation == 'get':
            cached_value = self._get_item(cache_key)
            if cached_value is not None:
                context['cache_hit'] = True
                logger.info(f"CacheManagerNode '{self.node_name}': Cache HIT for key '{cache_key}'.")
                return cached_value
            else:
                context['cache_hit'] = False
                logger.info(f"CacheManagerNode '{self.node_name}': Cache MISS for key '{cache_key}'. Returning original data.")
                return data

        elif cache_operation == 'set':
            self._set_item(cache_key, data)
            context['cache_hit'] = False  # A 'set' operation itself is not a cache 'hit'
            logger.info(f"CacheManagerNode '{self.node_name}': Explicitly SET item for key '{cache_key}'.")
            return data  # Return original data after setting it

        elif cache_operation == 'invalidate':
            self._invalidate_item(cache_key)
            context['cache_hit'] = False  # An 'invalidate' operation is not a cache 'hit'
            logger.info(f"CacheManagerNode '{self.node_name}': Explicitly INVALIDATED item for key '{cache_key}'.")
            return data  # Return original data after invalidating

        elif cache_operation == 'passthrough':
            logger.debug(f"CacheManagerNode '{self.node_name}': Passthrough operation. Bypassing cache logic.")
            context['cache_hit'] = False
            return data

        else:
            logger.error(
                f"CacheManagerNode '{self.node_name}': Received unknown cache_operation '{cache_operation}'. "
                "Returning original data and marking as no cache hit."
            )
            context['cache_hit'] = False
            return data