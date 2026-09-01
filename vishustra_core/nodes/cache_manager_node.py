import logging
import time
from typing import Any, Dict, Optional

# Assuming this path from the project context
from vishustra_core.nodes.base_node import BaseNode 

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A Vishustra processing node that manages a shared in-memory cache
    within the orchestration context.

    This node provides functionality to perform standard cache operations:
    GET, SET, DELETE, and CLEAR. It also supports Time-To-Live (TTL)
    for cached items. The actual cache data structure is expected to be
    a dictionary stored under the 'cache_store' key within the
    processing `context` dictionary, allowing for a shared cache across nodes
    within an orchestration.
    """

    def __init__(self):
        """
        Initializes the CacheManagerNode. This node does not maintain
        an instance-level cache; it operates exclusively on the cache
        provided or initialized within the `context`.
        """
        logger.debug("CacheManagerNode initialized.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "CacheManager"

    def _get_expiry_timestamp(self, ttl_seconds: Optional[int]) -> Optional[float]:
        """
        Calculates the absolute expiry timestamp based on the given Time-To-Live
        in seconds.

        Args:
            ttl_seconds (Optional[int]): The Time-To-Live in seconds.
                                         If None or non-positive, the item
                                         is considered to never expire.

        Returns:
            Optional[float]: The Unix timestamp at which the item expires,
                             or None if it should not expire.
        """
        if ttl_seconds is None or ttl_seconds <= 0:
            return None  # No expiry
        return time.time() + ttl_seconds

    def _is_expired(self, expiry_timestamp: Optional[float]) -> bool:
        """
        Checks if a cached item, identified by its expiry timestamp, has expired.

        Args:
            expiry_timestamp (Optional[float]): The Unix timestamp of expiry,
                                                or None if the item does not expire.

        Returns:
            bool: True if the item has expired, False otherwise.
        """
        if expiry_timestamp is None:
            return False  # No expiry specified, so never expires
        return time.time() >= expiry_timestamp

    def process(self, data: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """
        Executes cache operations on the shared 'cache_store' within the context.

        The `data` input dictionary must specify the operation and its parameters.
        Supported operations are:

        - **SET**: `{"operation": "SET", "key": Any, "value": Any, "ttl": Optional[int]}`
          Stores `value` under `key`. `ttl` is optional (in seconds).
          Returns `True` on success.
        - **GET**: `{"operation": "GET", "key": Any}`
          Retrieves the `value` associated with `key`.
          Returns the cached `value` or `None` if not found or expired.
        - **DELETE**: `{"operation": "DELETE", "key": Any}`
          Removes the entry associated with `key`.
          Returns `True` if deleted, `False` if the key was not found.
        - **CLEAR**: `{"operation": "CLEAR"}`
          Empties the entire cache.
          Returns `True`.

        Args:
            data (Dict[str, Any]): A dictionary defining the cache operation
                                    and its arguments.
            context (Dict[str, Any]): The shared processing context. This dictionary
                                      is expected to contain, or will be initialized
                                      with, a mutable 'cache_store' dictionary.

        Returns:
            Any: The result of the cache operation, which can be the retrieved
                 value, a boolean indicating success, or None.

        Raises:
            ValueError: If the 'operation' is invalid or mandatory parameters
                        are missing for a specific operation.
            Exception: For any other unexpected errors during cache processing.
        """
        # Ensure 'cache_store' exists in the context and is a dictionary
        if "cache_store" not in context or not isinstance(context["cache_store"], dict):
            logger.info("Initializing 'cache_store' in context as it was not found or not a dict.")
            context["cache_store"] = {}
        
        cache_store: Dict[Any, Dict[str, Any]] = context["cache_store"]

        operation: str = data.get("operation", "").upper()
        key: Any = data.get("key")
        value: Any = data.get("value") # Note: 'value' might genuinely be None
        ttl: Optional[int] = data.get("ttl")

        logger.debug(f"CacheManagerNode received operation: {operation} for key: {key if key is not None else 'N/A'}")

        try:
            if operation == "SET":
                if key is None:
                    raise ValueError("SET operation requires a 'key'.")
                # Explicitly check for 'value' key existence as its content could be None
                if "value" not in data:
                    raise ValueError("SET operation requires a 'value' parameter.")
                
                expiry = self._get_expiry_timestamp(ttl)
                cache_store[key] = {"value": value, "expiry": expiry}
                logger.info(f"Set cache entry for key '{key}'. Expires: {time.ctime(expiry) if expiry else 'Never'}")
                return True

            elif operation == "GET":
                if key is None:
                    raise ValueError("GET operation requires a 'key'.")
                
                entry = cache_store.get(key)
                if entry is None:
                    logger.debug(f"Cache miss for key '{key}'.")
                    return None
                
                if self._is_expired(entry.get("expiry")):
                    logger.info(f"Cache entry for key '{key}' has expired. Deleting it from cache.")
                    del cache_store[key]
                    return None
                
                logger.debug(f"Cache hit for key '{key}'.")
                return entry["value"]

            elif operation == "DELETE":
                if key is None:
                    raise ValueError("DELETE operation requires a 'key'.")
                
                if key in cache_store:
                    del cache_store[key]
                    logger.info(f"Deleted cache entry for key '{key}'.")
                    return True
                else:
                    logger.debug(f"Attempted to delete non-existent cache entry for key '{key}'.")
                    return False

            elif operation == "CLEAR":
                initial_size = len(cache_store)
                cache_store.clear()
                logger.info(f"Cleared cache. Removed {initial_size} entries.")
                return True

            else:
                raise ValueError(f"Unknown or unsupported cache operation: '{operation}'. "
                                 "Expected 'GET', 'SET', 'DELETE', or 'CLEAR'.")

        except ValueError as e:
            logger.error(f"Validation error in CacheManagerNode for operation '{operation}' (key: '{key}'): {e}")
            raise # Re-raise to propagate the error up the chain for handling by the orchestration framework
        except Exception as e:
            logger.error(f"An unexpected error occurred in CacheManagerNode during operation '{operation}' "
                         f"for key '{key}': {e}", exc_info=True)
            raise # Re-raise unexpected errors to ensure they are not silently swallowed