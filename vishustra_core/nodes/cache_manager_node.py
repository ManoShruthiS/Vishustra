import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Protocol

# Assuming BaseNode is available at this path as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

# --- Cache Client Protocol and In-Memory Implementation ---
class CacheClient(Protocol):
    """
    Protocol for a cache client interface.
    Ensures compatibility with various cache implementations (e.g., Redis, Memcached, in-memory).
    """
    def get(self, key: str) -> Optional[Any]:
        """Retrieves a value from the cache by its key."""
        ...

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Stores a value in the cache with an optional time-to-live."""
        ...

    def delete(self, key: str) -> None:
        """Removes a value from the cache by its key."""
        ...

class InMemoryCacheClient:
    """
    A simple, thread-safe in-memory cache client for demonstration and testing purposes.
    Supports basic get, set, and delete operations with optional TTL.

    Note: For production environments, consider external caching solutions like Redis
    or Memcached for better performance, persistence, and distributed capabilities.
    """
    def __init__(self):
        # Stores {key: {"value": ..., "expiry_time": datetime_obj or None}}
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._logger = logging.getLogger(self.__class__.__name__)
        # In a real multi-threaded application, a threading.Lock would be essential
        # for `_cache` operations to prevent race conditions. For this single-node
        # example, we'll omit it for simplicity, assuming higher-level concurrency
        # management or single-threaded node execution.

    def get(self, key: str) -> Optional[Any]:
        """Retrieves a value from the in-memory cache."""
        entry = self._cache.get(key)
        if entry:
            expiry_time = entry.get("expiry_time")
            if expiry_time is None or datetime.now() < expiry_time:
                self._logger.debug(f"Cache hit for key: '{key}'")
                return entry["value"]
            else:
                self._logger.debug(f"Cache entry for key: '{key}' expired. Removing.")
                del self._cache[key] # Clean up expired entry
        self._logger.debug(f"Cache miss for key: '{key}'")
        return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Stores a value in the in-memory cache with an optional TTL."""
        expiry_time = None
        if ttl_seconds is not None and ttl_seconds > 0:
            expiry_time = datetime.now() + timedelta(seconds=ttl_seconds)
        self._cache[key] = {"value": value, "expiry_time": expiry_time}
        self._logger.debug(f"Cache set for key: '{key}' with TTL: {ttl_seconds}s")

    def delete(self, key: str) -> None:
        """Removes a value from the in-memory cache."""
        if key in self._cache:
            del self._cache[key]
            self._logger.debug(f"Cache deleted for key: '{key}'")
        else:
            self._logger.debug(f"Attempted to delete non-existent cache key: '{key}'")

# --- CacheManagerNode Implementation ---
class CacheManagerNode(BaseNode):
    """
    A Vishustra node responsible for interacting with a caching system.
    It can perform 'get', 'set', or 'invalidate' operations based on parameters
    provided in the `context` dictionary.

    This node provides a flexible interface for integrating caching into
    LLM orchestration workflows, allowing for result memoization, state sharing,
    and reducing redundant computations.

    Context parameters consumed by this node:
    - `cache_key` (str): The unique identifier for the cache entry. (Required for all actions)
    - `cache_action` (str): The desired cache operation: 'get', 'set', 'invalidate'.
                            Defaults to 'passthrough' if not specified or invalid.
    - `cache_ttl` (int, optional): Time-to-live in seconds for 'set' operations.
                                   If omitted or non-positive, the entry may not expire.

    Context parameters updated by this node:
    - `vishustra_cache_status` (str): Reports the outcome of the cache operation.
      Possible values: 'HIT', 'MISS', 'SET', 'INVALIDATED', 'ERROR', 'NO_ACTION'.
    - `vishustra_cached_value` (Any, optional): The value retrieved from the cache
      if a 'get' action results in a cache hit.
    """

    def __init__(self, cache_client: Optional[CacheClient] = None):
        """
        Initializes the CacheManagerNode.

        Args:
            cache_client (Optional[CacheClient]): An optional cache client implementation
                                                  conforming to the `CacheClient` protocol.
                                                  If None, an `InMemoryCacheClient` is used by default.
        """
        self._cache_client: CacheClient = cache_client if cache_client is not None else InMemoryCacheClient()
        logger.info(f"{self.node_name} initialized with cache client: {type(self._cache_client).__name__}")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "CacheManagerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by interacting with the configured cache client
        based on `context` parameters.

        Args:
            data (Any): The input data to be processed. This data may be stored
                        in the cache, or passed through, or represent a key to retrieve.
            context (Dict[str, Any]): The shared context dictionary containing
                                       parameters for cache operations and
                                       where cache status will be reported.

        Returns:
            Any: If the `cache_action` is 'get' and results in a cache hit,
                 the *cached value* is returned. For all other actions or a
                 cache miss, the *original `data` input* to this node is returned,
                 allowing subsequent nodes to continue processing or compute the value.
        """
        cache_key = context.get("cache_key")
        cache_action = context.get("cache_action", "passthrough").lower()
        cache_ttl = context.get("cache_ttl")

        # Initialize cache status in context to provide clear visibility
        context["vishustra_cache_status"] = "NO_ACTION"
        context.pop("vishustra_cached_value", None) # Ensure previous cached value is cleared

        if cache_action != "passthrough" and not isinstance(cache_key, str):
            logger.error(f"{self.node_name}: 'cache_key' must be a string for action '{cache_action}'. "
                         f"Received type: {type(cache_key).__name__}. No cache operation performed.")
            context["vishustra_cache_status"] = "ERROR"
            return data

        try:
            if cache_action == "get":
                cached_value = self._cache_client.get(cache_key)
                if cached_value is not None:
                    context["vishustra_cache_status"] = "HIT"
                    context["vishustra_cached_value"] = cached_value
                    logger.debug(f"{self.node_name}: Cache hit for key '{cache_key}'. Returning cached value.")
                    return cached_value # On hit, return the cached data

                context["vishustra_cache_status"] = "MISS"
                logger.debug(f"{self.node_name}: Cache miss for key '{cache_key}'. Passing through original data.")
                return data # On miss, pass through original data for computation

            elif cache_action == "set":
                if cache_ttl is not None and not isinstance(cache_ttl, int):
                    logger.warning(f"{self.node_name}: 'cache_ttl' must be an integer for set operations. "
                                   f"Received type: {type(cache_ttl).__name__}. Proceeding with no TTL (infinite).")
                    cache_ttl = None
                
                self._cache_client.set(cache_key, data, cache_ttl)
                context["vishustra_cache_status"] = "SET"
                logger.debug(f"{self.node_name}: Cache set for key '{cache_key}'. Passing through original data.")
                return data # After setting, pass through original data

            elif cache_action == "invalidate":
                self._cache_client.delete(cache_key)
                context["vishustra_cache_status"] = "INVALIDATED"
                logger.debug(f"{self.node_name}: Cache invalidated for key '{cache_key}'. Passing through original data.")
                return data # After invalidating, pass through original data

            elif cache_action == "passthrough":
                logger.debug(f"{self.node_name}: No explicit cache action specified or required parameters missing. "
                             f"Passing data through unchanged.")
                return data

            else:
                logger.warning(f"{self.node_name}: Unknown or unsupported 'cache_action': '{cache_action}'. "
                               f"Supported actions: 'get', 'set', 'invalidate'. Passing data through.")
                context["vishustra_cache_status"] = "ERROR"
                return data

        except Exception as e:
            logger.exception(f"{self.node_name}: An unexpected error occurred during cache operation for key '{cache_key}': {e}")
            context["vishustra_cache_status"] = "ERROR"
            # In case of any error, ensure the original data continues through the pipeline
            return data
