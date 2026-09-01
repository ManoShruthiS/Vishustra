import logging
import time
from threading import Lock
from typing import Any, Dict, Tuple, Optional

# Assuming BaseNode will be imported from a specific path relative to the project root
# For development context, this path is `vishustra_core.nodes.base_node`
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A Vishustra processing node that provides versatile in-memory caching capabilities.
    It supports 'get', 'set', 'get_or_set', 'invalidate', and 'clear' operations
    on a thread-safe, time-aware cache.

    The node expects caching instructions via the 'context' dictionary.

    Context parameters:
    - 'cache_key' (str, required for most actions): The unique key for the cache entry.
                                                 Not required for 'clear' action.
    - 'cache_action' (str, optional, default='get_or_set'):
        Specifies the caching operation to perform:
        - 'get': Attempts to retrieve data. Returns cached data or None if not found/expired.
                 The `data` input to process() is ignored.
        - 'set': Stores the `data` provided to process(). Returns the stored data.
        - 'get_or_set': Tries to 'get'. If a cache miss or expired entry occurs,
                        it 'sets' the `data` provided to process().
                        Returns the cached data or the newly set data.
        - 'invalidate': Removes a specific entry identified by 'cache_key'.
                        Returns True if removed, False otherwise.
                        The `data` input to process() is ignored.
        - 'clear': Clears the entire cache. Returns True.
                   The `data` input to process() and 'cache_key' are ignored.
    - 'cache_ttl' (Union[int, float], optional): Time-to-live for the cache entry in seconds.
                                                 Only relevant for 'set' or 'get_or_set' actions.
                                                 If not provided, the entry will not expire naturally.
    """

    def __init__(self):
        """Initializes the in-memory cache storage and a re-entrant lock for thread safety."""
        # _cache stores (value, expiry_timestamp)
        # expiry_timestamp is None if no TTL is set
        self._cache: Dict[str, Tuple[Any, Optional[float]]] = {}
        self._cache_lock = Lock()
        logger.debug(f"{self.node_name} node initialized.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "CacheManager"

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """
        Retrieves an item from the cache if it exists and is not expired.
        Returns the cached value or None on a miss or expiry.
        """
        with self._cache_lock:
            cached_item = self._cache.get(key)
            if cached_item:
                value, expiry_time = cached_item
                if expiry_time is None or time.monotonic() < expiry_time:
                    logger.debug(f"Cache hit for key: '{key}'.")
                    return value
                else:
                    logger.debug(f"Cache miss for key: '{key}' (entry expired). Removing expired item.")
                    del self._cache[key]  # Clean up expired item
            logger.debug(f"Cache miss for key: '{key}' (not found).")
            return None

    def _set_to_cache(self, key: str, value: Any, ttl: Optional[float] = None) -> Any:
        """
        Stores an item in the cache with an optional Time-To-Live (TTL).
        Returns the value that was stored.
        """
        expiry_time = time.monotonic() + ttl if ttl is not None else None
        with self._cache_lock:
            self._cache[key] = (value, expiry_time)
        logger.debug(f"Cache set for key: '{key}', TTL: {ttl if ttl is not None else 'infinite'}s.")
        return value

    def _invalidate_cache_entry(self, key: str) -> bool:
        """
        Invalidates (removes) a specific entry from the cache.
        Returns True if the entry was found and removed, False otherwise.
        """
        with self._cache_lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"Cache entry invalidated for key: '{key}'.")
                return True
            logger.debug(f"Attempted to invalidate non-existent cache key: '{key}'.")
            return False

    def _clear_cache(self) -> bool:
        """
        Clears all entries from the cache.
        Returns True upon successful clearing.
        """
        with self._cache_lock:
            self._cache.clear()
        logger.info("CacheManagerNode cache cleared.")
        return True

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes a caching operation based on the provided `context`.

        Args:
            data (Any): The payload to be cached. This is used by 'set' and 'get_or_set'
                        actions when a value needs to be stored. Ignored by other actions.
            context (Dict[str, Any]): A dictionary containing 'cache_key', 'cache_action',
                                     and optionally 'cache_ttl' to guide the cache operation.

        Returns:
            Any: The result of the cache operation, which varies by action:
                 - 'get': The cached data, or None if not found/expired.
                 - 'set': The data that was just stored.
                 - 'get_or_set': The cached data, or the newly stored data if a miss occurred.
                 - 'invalidate': True if the key was found and removed, False otherwise.
                 - 'clear': True.

        Raises:
            ValueError: If 'cache_key' is missing for actions that require it,
                        or if 'cache_action' is unrecognized.
            TypeError: If 'cache_ttl' is not a valid numeric type or is negative.
        """
        cache_action = context.get("cache_action", "get_or_set").lower()
        cache_ttl = context.get("cache_ttl")

        if cache_ttl is not None:
            if not isinstance(cache_ttl, (int, float)) or cache_ttl < 0:
                logger.error(f"Invalid 'cache_ttl' provided in context: '{cache_ttl}'. Must be a non-negative number.")
                raise TypeError(f"Invalid 'cache_ttl' type or value. Expected non-negative number, got {type(cache_ttl).__name__}.")

        if cache_action == "clear":
            return self._clear_cache()

        # For all other actions, a 'cache_key' is mandatory.
        cache_key = context.get("cache_key")
        if not isinstance(cache_key, str) or not cache_key:
            logger.error(f"Context missing required 'cache_key' or it's not a valid string for action '{cache_action}'.")
            raise ValueError(f"{self.node_name} requires a valid 'cache_key' (string) in the context for action '{cache_action}'.")

        if cache_action == "get":
            return self._get_from_cache(cache_key)
        elif cache_action == "set":
            if data is None:
                logger.warning(f"Attempting to 'set' None data for key '{cache_key}'. While allowed, ensure this is intended.")
            return self._set_to_cache(cache_key, data, cache_ttl)
        elif cache_action == "get_or_set":
            cached_value = self._get_from_cache(cache_key)
            if cached_value is not None:
                return cached_value
            # Cache miss or expired, proceed to set
            if data is None:
                logger.warning(f"Cache miss for key '{cache_key}', attempting to 'set' None data. Ensure this is intended.")
            return self._set_to_cache(cache_key, data, cache_ttl)
        elif cache_action == "invalidate":
            return self._invalidate_cache_entry(cache_key)
        else:
            logger.error(f"Invalid 'cache_action' specified in context: '{cache_action}'.")
            raise ValueError(f"Invalid 'cache_action': '{cache_action}'. Expected one of 'get', 'set', 'get_or_set', 'invalidate', 'clear'.")

