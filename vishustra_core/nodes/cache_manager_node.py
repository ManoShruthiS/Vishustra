import logging
import time
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A processing node designed for intelligent management of a shared cache store
    within the Vishustra execution context.

    This node provides robust mechanisms for caching, retrieving, and invalidating
    data, which is crucial for optimizing performance in LLM orchestration workflows.
    It supports operations such as 'get', 'set', 'clear', and 'invalidate_key',
    enabling other nodes to effectively leverage a common, in-memory cache.

    The cache store is maintained as a dictionary within the global `context`
    object, ensuring consistency and accessibility across various stages of
    an orchestration pipeline. Each cache entry includes a value and an optional
    expiry timestamp, allowing for time-to-live (TTL) based cache management.
    """

    # A well-defined sentinel key to store the cache in the context,
    # minimizing potential collisions with user-defined context keys.
    _CACHE_CONTEXT_KEY = "_vishustra_shared_cache_store"
    _DEFAULT_TTL_SECONDS = 300  # Default Time-To-Live for new cache entries (5 minutes).

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "CacheManagerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes cache operations based on the provided input `data`.

        The `data` dictionary dictates the specific cache action to perform
        and its parameters. The cache itself is managed within the `context`.

        Expected `data` structure for operations:
        - **'get'**: {"operation": "get", "key": str}
            Retrieves a cached value. Returns the value if found and valid, else None.
        - **'set'**: {"operation": "set", "key": str, "value": Any, "ttl": Optional[int]}
            Stores a value in the cache. `ttl` (in seconds) is optional; if omitted
            or None, `_DEFAULT_TTL_SECONDS` is used. If `ttl=0` (or effectively infinite),
            the entry will not expire naturally. Returns True on success.
        - **'invalidate_key'**: {"operation": "invalidate_key", "key": str}
            Removes a specific key-value pair from the cache, regardless of its expiry.
            Returns True if the key was found and removed, False otherwise.
        - **'clear'**: {"operation": "clear"}
            Empties the entire cache store. Returns True on success.

        Each cache entry is stored as a tuple: `(value, expiry_timestamp)`.
        `expiry_timestamp` is `None` for entries that do not expire.

        Args:
            data: A dictionary containing the cache operation details.
            context: The shared mutable context dictionary, which houses the cache store.

        Returns:
            Any: The result of the cache operation, varying by operation type:
                 - For 'get': The cached value (Any) or None if not found/expired.
                 - For 'set': `True` on successful caching.
                 - For 'invalidate_key': `True` if the key was present and removed, `False` otherwise.
                 - For 'clear': `True` on successful clearing.

        Raises:
            ValueError: If the `data` input is malformed, missing required fields,
                        or specifies an unknown/invalid operation or parameter.
            Exception: Re-raises any unexpected errors encountered during processing
                       after logging them for diagnostic purposes.
        """
        if not isinstance(data, dict):
            logger.error("CacheManagerNode received non-dictionary input data: %s", type(data))
            raise ValueError("Invalid input: 'data' must be a dictionary specifying the cache operation.")

        operation = data.get("operation")
        cache_key = data.get("key")
        value_to_cache = data.get("value")
        # Use default TTL if 'ttl' is not provided in data, else use provided value
        ttl_seconds = data.get("ttl", self._DEFAULT_TTL_SECONDS)

        # Initialize cache store if it does not yet exist in the context
        if self._CACHE_CONTEXT_KEY not in context:
            context[self._CACHE_CONTEXT_KEY] = {}
            logger.debug("Initialized shared cache store in context under key '%s'.", self._CACHE_CONTEXT_KEY)
        
        # Type hint for clarity; cache_store is a mutable reference to the dictionary in context
        cache_store: Dict[str, Any] = context[self._CACHE_CONTEXT_KEY]

        current_time = time.time()

        try:
            if operation == "get":
                if not isinstance(cache_key, str):
                    logger.warning("CacheManagerNode 'get' operation received non-string key '%s'. Returning None.", cache_key)
                    return None
                
                entry = cache_store.get(cache_key)
                if entry:
                    cached_value, expiry_time = entry
                    if expiry_time is None or current_time < expiry_time:
                        logger.debug("Cache hit for key '%s'.", cache_key)
                        return cached_value
                    else:
                        # Entry expired, remove it to keep cache clean
                        del cache_store[cache_key]
                        logger.info("Cache entry for key '%s' expired and was removed.", cache_key)
                logger.debug("Cache miss for key '%s'.", cache_key)
                return None

            elif operation == "set":
                if not isinstance(cache_key, str):
                    logger.error("CacheManagerNode 'set' operation requires a string 'key', received type: %s", type(cache_key))
                    raise ValueError("Invalid input: 'set' operation requires a string 'key'.")
                
                # Validate TTL
                if ttl_seconds is not None and (not isinstance(ttl_seconds, int) or ttl_seconds < 0):
                    logger.error("CacheManagerNode 'set' operation received invalid 'ttl': %s. Must be a non-negative integer.", ttl_seconds)
                    raise ValueError("Invalid input: 'set' operation 'ttl' must be a non-negative integer.")

                # Calculate expiry time. If ttl_seconds is 0 or None, it means infinite cache.
                expiry_time = current_time + ttl_seconds if ttl_seconds is not None and ttl_seconds > 0 else None
                
                cache_store[cache_key] = (value_to_cache, expiry_time)
                logger.info("Cached value for key '%s' with TTL of %s seconds (expires: %s).",
                            cache_key, ttl_seconds if ttl_seconds is not None else "infinite",
                            time.ctime(expiry_time) if expiry_time else "Never")
                return True

            elif operation == "invalidate_key":
                if not isinstance(cache_key, str):
                    logger.error("CacheManagerNode 'invalidate_key' operation requires a string 'key', received type: %s", type(cache_key))
                    raise ValueError("Invalid input: 'invalidate_key' operation requires a string 'key'.")
                
                if cache_key in cache_store:
                    del cache_store[cache_key]
                    logger.info("Invalidated cache entry for key '%s'.", cache_key)
                    return True
                else:
                    logger.debug("Attempted to invalidate non-existent cache key '%s'. No action taken.", cache_key)
                    return False

            elif operation == "clear":
                cache_store.clear()
                logger.info("Cleared all entries from the shared cache.")
                return True

            else:
                logger.error("CacheManagerNode received an unknown or unsupported operation: '%s'", operation)
                raise ValueError(f"Unknown cache operation: '{operation}'. "
                                 "Expected one of 'get', 'set', 'clear', or 'invalidate_key'.")

        except Exception as e:
            # Catch all other unexpected errors to log before re-raising
            logger.error("An unexpected error occurred during cache operation '%s' for key '%s': %s",
                         operation, cache_key if cache_key else "N/A", e, exc_info=True)
            raise # Re-raise the original exception