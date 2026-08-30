
import logging
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A processing node for managing an in-memory, LRU (Least Recently Used) cache
    with optional Time-To-Live (TTL) functionality.

    This node provides 'get', 'set', 'invalidate', and 'clear' operations
    driven by the `context` dictionary.
    """

    def __init__(self, capacity: int = 128, ttl_seconds: Optional[int] = None):
        """
        Initializes the CacheManagerNode with a specified capacity and optional TTL.
        The cache uses an LRU eviction policy if capacity is exceeded.
        If ttl_seconds is provided, entries expire after that duration.

        Args:
            capacity (int): The maximum number of items the cache can hold. Must be positive.
                            Defaults to 128.
            ttl_seconds (Optional[int]): Time-to-live for cache entries in seconds.
                                         If None, entries do not expire.
        Raises:
            ValueError: If cache capacity is not a positive integer.
        """
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("Cache capacity must be a positive integer.")

        # Cache stores (value, timestamp) tuples for TTL management.
        # OrderedDict maintains insertion order, allowing LRU eviction by popping the first item.
        self._cache: OrderedDict[Any, Tuple[Any, float]] = OrderedDict()
        self._capacity = capacity
        self._ttl_seconds = ttl_seconds
        logger.info(
            f"CacheManagerNode initialized with capacity={capacity}, "
            f"ttl_seconds={ttl_seconds if ttl_seconds is not None else 'infinite'}."
        )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "CacheManagerNode"

    def _is_expired(self, timestamp: float) -> bool:
        """Checks if a cache entry has expired based on its timestamp and TTL."""
        if self._ttl_seconds is None:
            return False
        return (time.time() - timestamp) > self._ttl_seconds

    def _cleanup_expired_entries(self) -> None:
        """Removes expired entries from the cache to keep it fresh."""
        keys_to_remove = []
        # Iterate over a copy of items to allow deletion during iteration
        for key, (_, timestamp) in list(self._cache.items()):
            if self._is_expired(timestamp):
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self._cache[key]
        if keys_to_remove:
            logger.debug(f"Cleaned up {len(keys_to_remove)} expired cache entries.")

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Manages cache operations based on the provided data and context.
        The `data` input's role depends on the `cache_action`.

        Expected `context` keys:
        - 'cache_action': str (required) - Specifies the operation: 'get', 'set', 'invalidate', 'clear'.
        - 'cache_key': Any (required for 'get', 'set', 'invalidate') - The key for the cache operation.
                       If not provided for 'get' or 'invalidate', `data` will be used as the key.
        - 'cache_value': Any (optional for 'set') - The value to store. If not provided,
                         `data` will be used as the value for 'set' operations.

        Args:
            data (Any): The primary input data. Its specific role varies by `cache_action`:
                        - 'get': Used as `cache_key` if `context['cache_key']` is absent.
                        - 'set': Used as `cache_value` if `context['cache_value']` is absent.
                        - 'invalidate': Used as `cache_key` if `context['cache_key']` is absent.
                        - 'clear': Ignored.
            context (Dict[str, Any]): A dictionary containing parameters for the cache operation.

        Returns:
            Any: The result of the cache operation:
                 - 'get': The cached value if found and not expired, otherwise `None`.
                 - 'set': The value that was stored.
                 - 'invalidate': `None`.
                 - 'clear': `None`.

        Raises:
            ValueError: If 'cache_action' is missing or unknown.
            KeyError: If 'cache_key' is required but not provided in context (e.g., for 'set')
                      and `data` cannot implicitly serve as the key.
        """
        self._cleanup_expired_entries() # Perform cleanup before any operation

        cache_action = context.get('cache_action')
        if not cache_action:
            logger.error("Missing 'cache_action' in context for CacheManagerNode.")
            raise ValueError("Cache action must be specified in context.")

        if cache_action == 'get':
            cache_key = context.get('cache_key', data) # Use 'data' as key if 'cache_key' not explicit
            if cache_key in self._cache:
                value, timestamp = self._cache[cache_key]
                if not self._is_expired(timestamp):
                    # Move to end to signify recent use (LRU)
                    self._cache.move_to_end(cache_key)
                    logger.debug(f"Cache hit for key: '{cache_key}'.")
                    return value
                else:
                    del self._cache[cache_key] # Remove expired entry
                    logger.debug(f"Cache miss (expired) for key: '{cache_key}'.")
            logger.debug(f"Cache miss for key: '{cache_key}'.")
            return None

        elif cache_action == 'set':
            cache_key = context.get('cache_key')
            if cache_key is None:
                # 'data' is a suitable default for value, but not always for key in a set.
                # Enforce explicit 'cache_key' for clarity when setting.
                logger.error("Missing 'cache_key' in context for 'set' action.")
                raise KeyError("A 'cache_key' must be specified in context for 'set' action.")

            cache_value = context.get('cache_value', data) # Use 'data' as value if 'cache_value' not explicit
            self._cache[cache_key] = (cache_value, time.time())
            self._cache.move_to_end(cache_key) # Mark as recently used

            # Evict LRU entry if capacity exceeded
            while len(self._cache) > self._capacity:
                lru_key = next(iter(self._cache)) # Get the first (LRU) key
                del self._cache[lru_key]
                logger.debug(f"Cache evicted LRU entry for key: '{lru_key}'.")

            logger.debug(f"Cache set for key: '{cache_key}'.")
            return cache_value

        elif cache_action == 'invalidate':
            cache_key = context.get('cache_key', data) # Use 'data' as key if 'cache_key' not explicit
            if cache_key in self._cache:
                del self._cache[cache_key]
                logger.debug(f"Cache invalidated for key: '{cache_key}'.")
            else:
                logger.debug(f"Attempted to invalidate non-existent key: '{cache_key}'.")
            return None

        elif cache_action == 'clear':
            self._cache.clear()
            logger.debug("Cache cleared.")
            return None

        else:
            logger.error(f"Unknown cache action: '{cache_action}'.")
            raise ValueError(f"Unknown cache action: '{cache_action}'. "
                             "Expected 'get', 'set', 'invalidate', or 'clear'.")

