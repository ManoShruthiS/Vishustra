
import logging
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManager(BaseNode):
    """
    A processing node for managing an in-memory cache within the Vishustra pipeline.
    This node can perform cache lookups and store data, based on control parameters
    provided in the `context` dictionary.

    The cache maintains entries with a time-to-live (TTL) and employs a basic
    Least Recently Used (LRU) eviction strategy when the maximum capacity is reached.

    Context Parameters Understood:
    ------------------------------
    - 'cache_key': (Required for most operations) A unique, hashable key for the cache entry.
    - 'cache_action': (Optional, default 'lookup') Specifies the desired operation:
        - 'lookup': Attempt to retrieve data from the cache.
        - 'store': Store data into the cache.
    - 'cache_force_refresh': (Optional, bool, default False for 'lookup' action)
        If True, the lookup operation will bypass the cache and always result in a miss,
        forcing downstream processing.
    - 'cache_ttl_seconds': (Optional, int, default 300 for 'store' action)
        The time-to-live in seconds for the stored cache entry.
    - 'cache_value_to_store': (Optional, Any, for 'store' action)
        The explicit value to store in the cache. If not provided, the `data`
        argument passed to the `process` method will be used as the value.

    Output Context Modifications:
    -----------------------------
    - 'cache_hit': (bool) Set to True if a 'lookup' action found a valid, non-expired entry.
    - 'cache_value': (Any) If 'cache_hit' is True, this contains the retrieved cached value.
    - 'cache_stored': (bool) Set to True if a 'store' action successfully saved data.
    """

    # Using OrderedDict to simulate a basic LRU for in-memory cache
    _cache: OrderedDict[Any, Dict[str, Any]]
    _max_cache_entries: int

    def __init__(self, max_cache_entries: int = 1000):
        """
        Initializes the CacheManager node.

        Args:
            max_cache_entries: The maximum number of entries the in-memory cache can hold.
                               If set to 0 or a negative number, caching effectively becomes
                               disabled (no new entries will be stored if already full).
        """
        if not isinstance(max_cache_entries, int) or max_cache_entries < 0:
            logger.warning(
                f"Invalid 'max_cache_entries' value: {max_cache_entries}. "
                "Defaulting to 1000 and ensuring it's not negative."
            )
            self._max_cache_entries = 1000
        else:
            self._max_cache_entries = max_cache_entries

        self._cache = OrderedDict()
        logger.info(f"CacheManager initialized with max_cache_entries={self._max_cache_entries}.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, performing either a cache lookup or a storage operation
        based on the 'cache_action' in the context.

        Args:
            data: The primary input for the current step. Its role depends on 'cache_action'.
                  - If 'lookup': `data` is typically the key to look up (or derived from).
                  - If 'store': `data` is typically the value that has just been processed
                                by a previous node and is now ready to be cached.
            context: A dictionary containing execution context and cache control parameters.

        Returns:
            - If 'lookup' and a valid cache hit: The retrieved cached value.
            - If 'lookup' and a cache miss/expired/forced refresh: The original `data`
              (often the key/request) to allow downstream processing to generate the value.
            - If 'store': The original `data` (the value being stored or the result
              that was passed through). The node primarily performs a side effect (storage).
        """
        cache_key = context.get('cache_key')
        cache_action = context.get('cache_action', 'lookup').lower()

        # Reset cache-related context flags for this operation
        context['cache_hit'] = False
        context.pop('cache_value', None)
        context['cache_stored'] = False

        if cache_key is None:
            logger.warning(
                "CacheManager received data without a 'cache_key' in context. "
                "Skipping cache operation and passing data through."
            )
            return data
        
        # Ensure cache_key is hashable
        try:
            hash(cache_key)
        except TypeError:
            logger.error(
                f"CacheManager received unhashable 'cache_key': {cache_key}. "
                "Skipping cache operation and passing data through."
            )
            return data

        if cache_action == 'lookup':
            return self._handle_lookup(cache_key, data, context)
        elif cache_action == 'store':
            return self._handle_store(cache_key, data, context)
        else:
            logger.error(
                f"CacheManager received unknown 'cache_action': '{cache_action}' for key '{cache_key}'. "
                "Skipping cache operation and passing data through."
            )
            return data

    def _handle_lookup(self, cache_key: Any, data: Any, context: Dict[str, Any]) -> Any:
        """Handles the cache lookup logic."""
        force_refresh = context.get('cache_force_refresh', False)

        if force_refresh:
            logger.debug(f"CacheManager: Forced refresh for key '{cache_key}'. Bypassing cache.")
            context['cache_hit'] = False
            return data

        entry = self._cache.get(cache_key)
        
        if entry:
            current_time = time.time()
            if current_time - entry['timestamp'] < entry['ttl']:
                # Cache hit and valid, move to end to mark as recently used
                self._cache.move_to_end(cache_key)
                logger.debug(f"CacheManager: Hit for key '{cache_key}'. Valid entry.")
                context['cache_hit'] = True
                context['cache_value'] = entry['value']
                return entry['value'] # Return the cached value
            else:
                logger.debug(f"CacheManager: Miss for key '{cache_key}'. Entry expired.")
                del self._cache[cache_key] # Evict expired entry
        else:
            logger.debug(f"CacheManager: Miss for key '{cache_key}'. No entry found.")

        context['cache_hit'] = False
        return data # Return original data (the key/request) for downstream processing

    def _handle_store(self, cache_key: Any, data: Any, context: Dict[str, Any]) -> Any:
        """Handles the cache storage logic."""
        # Prioritize 'cache_value_to_store' from context, otherwise use 'data'
        value_to_store = context.get('cache_value_to_store', data)
        ttl_seconds = context.get('cache_ttl_seconds', 300) # Default TTL to 5 minutes

        if not isinstance(ttl_seconds, (int, float)) or ttl_seconds <= 0:
            logger.warning(
                f"Invalid or non-positive 'cache_ttl_seconds' value '{ttl_seconds}' for key '{cache_key}'. "
                "Using default 300 seconds."
            )
            ttl_seconds = 300
        
        # Perform LRU eviction if cache is full and this key is not already present
        if self._max_cache_entries > 0 and len(self._cache) >= self._max_cache_entries and cache_key not in self._cache:
            # Evict the least recently used item (first in OrderedDict)
            oldest_key, _ = self._cache.popitem(last=False)
            logger.info(
                f"CacheManager: Cache full (max {self._max_cache_entries} entries). "
                f"Evicting '{oldest_key}' to make space for '{cache_key}'."
            )
        elif self._max_cache_entries == 0 and cache_key not in self._cache:
            logger.debug(
                f"CacheManager: Max cache entries set to 0. Not storing new key '{cache_key}'."
            )
            context['cache_stored'] = False
            return data

        self._cache[cache_key] = {
            'value': value_to_store,
            'timestamp': time.time(),
            'ttl': ttl_seconds
        }
        # Move to end to mark as recently used/updated
        self._cache.move_to_end(cache_key) 
        
        logger.info(f"CacheManager: Stored data for key '{cache_key}' with TTL {ttl_seconds}s.")
        context['cache_stored'] = True
        return data # Return original data (the value that was just stored) for downstream flow
