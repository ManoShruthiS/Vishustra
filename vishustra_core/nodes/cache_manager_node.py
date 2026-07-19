import logging
from typing import Any, Dict, Hashable, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class CacheManagerNode(BaseNode):
    """
    Manages an in-memory cache for Vishustra processing nodes.

    This node supports 'get', 'set', 'clear_key', and 'clear_all' operations
    based on instructions provided in the `context` dictionary. It provides
    efficient retrieval of previously processed data.
    """

    def __init__(self):
        super().__init__()
        self._cache: Dict[Hashable, Any] = {}
        logger.info("CacheManagerNode initialized with an empty in-memory cache.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Performs cache operations (get, set, clear_key, clear_all) based on
        the 'cache_operation' and 'cache_key' specified in the context.

        The `context` dictionary will be updated with `cache_status`, `cache_hit`,
        and `cache_result` to provide comprehensive feedback on the operation.

        Parameters:
            data (Any):
                - For 'set' operation: The value to be stored in the cache.
                - For 'get', 'clear_key', 'clear_all' operations: This parameter
                  is generally ignored by the CacheManagerNode itself, but it
                  is passed along as per the BaseNode interface.
            context (Dict[str, Any]): A dictionary containing operation details.
                - 'cache_operation' (str, required): Specifies the action to perform.
                  Must be one of "get", "set", "clear_key", or "clear_all".
                - 'cache_key' (Hashable, required for 'get', 'set', 'clear_key'):
                  The key associated with the cache entry.

        Returns:
            Any:
                - 'get' operation: The cached value if found, otherwise `None`.
                - 'set' operation: The value that was just stored in the cache.
                - 'clear_key' operation: `True` if the key was found and cleared,
                  `False` otherwise.
                - 'clear_all' operation: `True` upon successful clearing of the
                  entire cache.

        Raises:
            ValueError: If 'cache_operation' is missing or invalid, or if 'cache_key'
                        is missing for operations that require it.
            TypeError: If 'cache_key' is provided and is not hashable.
        """
        operation = context.get("cache_operation")
        cache_key = context.get("cache_key")

        # --- Input Validation ---
        if not operation:
            logger.error("CacheManagerNode received a request without 'cache_operation' in context.")
            raise ValueError("Missing 'cache_operation' in context. Expected 'get', 'set', 'clear_key', or 'clear_all'.")

        if operation in ["get", "set", "clear_key"] and cache_key is None:
            logger.error(f"CacheManagerNode operation '{operation}' requires 'cache_key' in context, but it was missing.")
            raise ValueError(f"Operation '{operation}' requires 'cache_key' in context.")

        if cache_key is not None:
            try:
                hash(cache_key)
            except TypeError as e:
                logger.error(f"CacheManagerNode received a non-hashable cache key: {cache_key} (Type: {type(cache_key).__name__})")
                raise TypeError(f"Cache key must be hashable. Encountered type: {type(cache_key).__name__}") from e

        # --- Initialize Context Feedback ---
        context["cache_status"] = "unknown"
        context["cache_hit"] = False
        context["cache_result"] = None

        # --- Perform Cache Operation ---
        if operation == "get":
            if cache_key in self._cache:
                value = self._cache[cache_key]
                context["cache_status"] = "hit"
                context["cache_hit"] = True
                context["cache_result"] = value
                logger.debug(f"Cache hit for key: '{cache_key}'")
                return value
            else:
                context["cache_status"] = "miss"
                context["cache_hit"] = False
                logger.debug(f"Cache miss for key: '{cache_key}'")
                return None

        elif operation == "set":
            self._cache[cache_key] = data
            context["cache_status"] = "set"
            context["cache_result"] = data
            logger.debug(f"Cache set for key: '{cache_key}' with value of type: {type(data).__name__}")
            return data

        elif operation == "clear_key":
            if cache_key in self._cache:
                del self._cache[cache_key]
                context["cache_status"] = "cleared_key"
                context["cache_result"] = True
                logger.debug(f"Cache entry cleared for key: '{cache_key}'")
                return True
            else:
                context["cache_status"] = "key_not_found"
                context["cache_result"] = False
                logger.warning(f"Attempted to clear non-existent cache key: '{cache_key}'")
                return False

        elif operation == "clear_all":
            self._cache.clear()
            context["cache_status"] = "cleared_all"
            context["cache_result"] = True
            logger.info("Entire cache has been cleared.")
            return True

        else:
            logger.error(f"CacheManagerNode received an unknown 'cache_operation': '{operation}'")
            raise ValueError(f"Unknown 'cache_operation': '{operation}'. Must be 'get', 'set', 'clear_key', or 'clear_all'.")