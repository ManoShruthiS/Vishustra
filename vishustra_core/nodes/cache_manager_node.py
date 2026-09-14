import logging
from collections import OrderedDict
from typing import Any, Dict, Optional

# Assuming BaseNode is available at this path in the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A Vishustra node that manages an in-memory cache.

    This node supports various cache operations such as 'get', 'set', 'evict', 'clear',
    and 'stats'. It maintains a fixed-size cache using a basic Least Recently Used (LRU)
    eviction policy.

    The 'process' method expects specific keys in the 'context' dictionary to define
    the desired cache operation.

    Context Parameters for 'process' method:
    --------------------------------------
    - 'cache_action' (str, required): The action to perform.
      Valid actions: 'get', 'set', 'evict', 'clear', 'stats'.
    - 'cache_key' (str, optional): The key for 'get', 'set', and 'evict' actions.
      Required for 'get', 'set', 'evict'. Not used for 'clear' or 'stats'.

    Data Parameter for 'process' method:
    ----------------------------------
    - 'data' (Any):
      - For 'set' action: The value to be stored in the cache.
      - For other actions ('get', 'evict', 'clear', 'stats'): Ignored.

    Returns from 'process' method:
    ----------------------------
    - 'get' action: The cached value associated with 'cache_key', or None if not found.
    - 'set' action: The value that was just set in the cache.
    - 'evict' action: True if the key was evicted, False otherwise.
    - 'clear' action: True.
    - 'stats' action: A dictionary containing cache hit, miss, and eviction statistics.

    Example Usage within a Vishustra pipeline:
    -----------------------------------------
    # In a pipeline definition or node configuration:
    # cache_node = CacheManagerNode(max_size=100)
    #
    # To set a value:
    # result = cache_node.process("my_value_to_cache", {"cache_action": "set", "cache_key": "my_data_id"})
    #
    # To get a value:
    # cached_data = cache_node.process(None, {"cache_action": "get", "cache_key": "my_data_id"})
    #
    # To get cache statistics:
    # current_stats = cache_node.process(None, {"cache_action": "stats"})
    """

    DEFAULT_MAX_SIZE = 100

    def __init__(self, max_size: int = DEFAULT_MAX_SIZE):
        """
        Initializes the CacheManagerNode.

        Args:
            max_size (int): The maximum number of items the cache can hold.
                            Defaults to 100. Must be a positive integer.
        """
        if not isinstance(max_size, int) or max_size <= 0:
            raise ValueError("max_size must be a positive integer.")

        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._max_size: int = max_size
        self._stats: Dict[str, int] = {'hits': 0, 'misses': 0, 'evictions': 0}
        logger.debug(f"CacheManagerNode initialized with max_size={self._max_size}")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data based on the specified cache action in the context.

        Args:
            data (Any): The data to be cached (for 'set' action) or ignored for others.
            context (Dict[str, Any]): A dictionary containing cache operation details.
                                       Must include 'cache_action' (str).
                                       May include 'cache_key' (str) for specific operations.

        Returns:
            Any: The result of the cache operation, varies by action.

        Raises:
            ValueError: If 'cache_action' is missing or invalid, or 'cache_key' is missing
                        or of incorrect type for actions that require it.
        """
        cache_action = context.get('cache_action')
        cache_key = context.get('cache_key')

        if not isinstance(cache_action, str):
            logger.error("Missing or invalid 'cache_action' in context for CacheManagerNode. Expected string.")
            raise ValueError("Context must specify 'cache_action' as a string.")

        if cache_action in ["get", "set", "evict"] and not isinstance(cache_key, str):
            logger.error(f"Missing or invalid 'cache_key' for action '{cache_action}'. Expected string.")
            raise ValueError(f"Action '{cache_action}' requires a 'cache_key' of type str.")

        try:
            if cache_action == "get":
                return self._handle_get(cache_key)
            elif cache_action == "set":
                return self._handle_set(cache_key, data)
            elif cache_action == "evict":
                return self._handle_evict(cache_key)
            elif cache_action == "clear":
                return self._handle_clear()
            elif cache_action == "stats":
                return self._handle_stats()
            else:
                logger.error(f"Unsupported 'cache_action' received: '{cache_action}'")
                raise ValueError(f"Unsupported 'cache_action': {cache_action}")
        except Exception as e:
            # Catching generic exceptions for robust error reporting
            operation_details = f"action='{cache_action}', key='{cache_key}'" if cache_key else f"action='{cache_action}'"
            logger.exception(f"An unexpected error occurred during cache operation ({operation_details}): {e}")
            raise

    def _handle_get(self, key: str) -> Optional[Any]:
        """Handles a 'get' cache action, retrieving a value by key."""
        if key in self._cache:
            value = self._cache[key]
            self._cache.move_to_end(key)  # Mark as recently used (LRU)
            self._stats['hits'] += 1
            logger.debug(f"Cache hit for key: '{key}'.")
            return value
        else:
            self._stats['misses'] += 1
            logger.debug(f"Cache miss for key: '{key}'.")
            return None

    def _handle_set(self, key: str, value: Any) -> Any:
        """Handles a 'set' cache action, storing a value with a key."""
        # Check if cache is full and key is not already present
        if key not in self._cache and len(self._cache) >= self._max_size:
            # Evict the least recently used item (first item in OrderedDict)
            oldest_key, _ = self._cache.popitem(last=False)
            self._stats['evictions'] += 1
            logger.info(f"Cache full ({self._max_size} items). Evicted least recently used key: '{oldest_key}'.")

        self._cache[key] = value
        self._cache.move_to_end(key) # Mark as recently used or move to end if updating existing key
        logger.info(f"Set cache for key: '{key}'. Cache size: {len(self._cache)}/{self._max_size}.")
        return value

    def _handle_evict(self, key: str) -> bool:
        """Handles an 'evict' cache action, removing a key-value pair."""
        if key in self._cache:
            del self._cache[key]
            logger.info(f"Evicted key from cache: '{key}'.")
            return True
        else:
            logger.warning(f"Attempted to evict non-existent key: '{key}'.")
            return False

    def _handle_clear(self) -> bool:
        """Handles a 'clear' cache action, emptying the cache."""
        self._cache.clear()
        self._stats = {'hits': 0, 'misses': 0, 'evictions': 0} # Reset stats on clear
        logger.info("Cache cleared and statistics reset.")
        return True

    def _handle_stats(self) -> Dict[str, int]:
        """Handles a 'stats' cache action, returning current cache statistics."""
        logger.debug("Retrieving cache statistics.")
        return self._stats