from vishustra_core.nodes.base_node import BaseNode
import logging
from typing import Any, Dict, Optional
from collections.abc import Hashable

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A processing node designed for managing a simple in-memory cache within the Vishustra framework.

    This node provides mechanisms for both retrieving and storing data in a cache.
    It expects a 'cache_operation' key in the context to determine the action.

    Supported Operations:
    - 'get': Attempts to retrieve a value from the cache. The cache key is derived from
             'context['cache_key']' or, if not present, from the 'data' argument itself
             (provided 'data' is hashable). If a cache hit occurs, the cached value is returned.
             If a miss occurs, a special sentinel `CacheManagerNode.CACHE_MISS` is returned.
    - 'set': Stores a value into the cache. The 'data' argument passed to `process` represents
             the value to be cached. The cache key for this operation *must* be explicitly
             provided in 'context['cache_key']'.

    This design allows the orchestration layer to first attempt a 'get', and upon a miss,
    perform a computation and then use a 'set' operation to store the result.
    """

    class CACHE_MISS:
        """Sentinel object to represent a cache miss, distinguishing it from `None` which could be a valid cached value."""
        def __repr__(self) -> str:
            return "<CACHE_MISS>"
    
    CACHE_MISS = CACHE_MISS()

    def __init__(self, initial_cache: Optional[Dict[Any, Any]] = None):
        """
        Initializes the CacheManagerNode.

        Args:
            initial_cache (Optional[Dict[Any, Any]]): An optional dictionary to pre-populate
                                                        the cache. Defaults to an empty cache.
        """
        self._cache: Dict[Any, Any] = initial_cache if initial_cache is not None else {}
        logger.debug(f"CacheManagerNode initialized with {len(self._cache)} initial items.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "CacheManager"

    def _derive_key_for_get(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Derives a cache key for a 'get' operation.
        Prioritizes 'cache_key' from context, then attempts to use 'data' if hashable.
        """
        if 'cache_key' in context:
            key = context['cache_key']
            logger.debug(f"Derived 'get' cache key from context: {key}")
            return key
        elif isinstance(data, Hashable):
            key = data
            logger.debug(f"Derived 'get' cache key from hashable data: {key}")
            return key
        else:
            raise ValueError(
                "Could not derive cache key for 'get' operation. "
                "Provide 'cache_key' in context or ensure input 'data' is hashable."
            )

    def _derive_key_for_set(self, context: Dict[str, Any]) -> Any:
        """
        Derives a cache key for a 'set' operation.
        Requires 'cache_key' to be explicitly present in the context.
        """
        if 'cache_key' in context:
            key = context['cache_key']
            logger.debug(f"Derived 'set' cache key from context: {key}")
            return key
        else:
            raise ValueError(
                "Cache key for 'set' operation must be explicitly provided in 'context['cache_key']'."
            )

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes the cache operation ('get' or 'set') based on the context.

        Args:
            data (Any):
                For 'get' operation: This is the query input from which a key may be derived.
                For 'set' operation: This is the actual value to be stored in the cache.
            context (Dict[str, Any]):
                A dictionary providing operational context. Must contain:
                - 'cache_operation' (str, optional): Specifies the action ('get' or 'set'). Defaults to 'get'.
                - 'cache_key' (Any, optional): The explicit key for cache operations.
                                               Required for 'set'. Optional for 'get' if 'data' is hashable.

        Returns:
            Any:
                - If 'get' operation results in a hit: The cached value.
                - If 'get' operation results in a miss: `CacheManagerNode.CACHE_MISS`.
                - If 'set' operation: The value that was just stored.

        Raises:
            ValueError: If an invalid 'cache_operation' is specified, or if a required
                        'cache_key' cannot be derived for the chosen operation.
        """
        operation = context.get('cache_operation', 'get').lower()
        
        if operation == 'get':
            try:
                cache_key = self._derive_key_for_get(data, context)
            except ValueError as e:
                logger.warning(f"Failed to derive cache key for 'get' operation. Returning CACHE_MISS. Error: {e}")
                return self.CACHE_MISS

            if cache_key in self._cache:
                value = self._cache[cache_key]
                logger.info(f"Cache hit for key '{cache_key}'.")
                return value
            else:
                logger.info(f"Cache miss for key '{cache_key}'.")
                return self.CACHE_MISS
        
        elif operation == 'set':
            try:
                cache_key = self._derive_key_for_set(context)
            except ValueError as e:
                error_msg = f"Cannot perform 'set' operation: {e}"
                logger.error(error_msg)
                raise ValueError(error_msg) from e

            self._cache[cache_key] = data
            logger.info(f"Cache set for key '{cache_key}'. Stored value type: {type(data).__name__}.")
            return data # Return the stored value
            
        else:
            error_msg = f"Invalid 'cache_operation': '{operation}'. Expected 'get' or 'set'."
            logger.error(error_msg)
            raise ValueError(error_msg)