import logging
from typing import Any, Dict, Optional

# Assuming vishustra_core.nodes.base_node is available in the Python path
# For local development/testing, you might need to adjust sys.path or use a relative import
# but for the framework context, this absolute import is standard.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A Vishustra processing node that provides a shared, in-memory cache
    within the orchestration context.

    This node enables other nodes in a Vishustra pipeline to efficiently
    store, retrieve, and invalidate data. The cache state is maintained
    within the `context` dictionary, making it accessible across node executions
    within the same orchestration run.

    Supported operations via the `process` method's `data` input:
    - 'retrieve': Fetches a value associated with a given key.
    - 'store': Stores a key-value pair in the cache.
    - 'invalidate': Removes a key and its associated value from the cache.
    """

    # Key used to store the cache dictionary within the context
    _CONTEXT_CACHE_KEY = "_vishustra_cache_manager_global_cache"

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to perform cache operations (retrieve, store, invalidate).

        Args:
            data (Any): A dictionary specifying the cache operation and arguments.
                        Expected format examples:
                        - {'action': 'retrieve', 'key': 'my_query_hash'}
                        - {'action': 'store', 'key': 'my_query_hash', 'value': {'llm_response': '...'}}
                        - {'action': 'invalidate', 'key': 'my_query_hash'}
            context (Dict[str, Any]): The shared context dictionary where the
                                      cache state is stored and managed.

        Returns:
            Any:
                - For 'retrieve': The cached value if found, otherwise None.
                - For 'store': The value that was successfully stored.
                - For 'invalidate': The key that was invalidated if it existed, otherwise None.

        Raises:
            ValueError: If the input `data` is malformed, missing required keys,
                        or specifies an unsupported cache action.
        """
        if not isinstance(data, dict):
            logger.error("CacheManagerNode received non-dictionary input for 'data': %s", type(data))
            raise ValueError(
                f"CacheManagerNode expects 'data' to be a dictionary, got {type(data).__name__}."
            )

        action = data.get('action')
        key = data.get('key')

        if not action or not key:
            logger.error("CacheManagerNode: Missing 'action' or 'key' in input data: %s", data)
            raise ValueError("Input data must contain 'action' and 'key' for cache operations.")

        # Initialize the global cache in the context if it doesn't exist
        if self._CONTEXT_CACHE_KEY not in context:
            context[self._CONTEXT_CACHE_KEY] = {}
            logger.debug("Initialized global cache in context for CacheManagerNode.")

        cache: Dict[Any, Any] = context[self._CONTEXT_CACHE_KEY]

        if action == 'retrieve':
            return self._handle_retrieve(cache, key)
        elif action == 'store':
            if 'value' not in data:
                logger.error("CacheManagerNode: 'store' action requires a 'value' key in data: %s", data)
                raise ValueError("'store' action requires a 'value' key in the input data.")
            value = data['value']
            return self._handle_store(cache, key, value)
        elif action == 'invalidate':
            return self._handle_invalidate(cache, key)
        else:
            logger.error("CacheManagerNode: Unknown action '%s' in input data: %s", action, data)
            raise ValueError(
                f"Unknown cache action: '{action}'. Supported actions are 'retrieve', 'store', 'invalidate'."
            )

    def _handle_retrieve(self, cache: Dict[Any, Any], key: Any) -> Optional[Any]:
        """Handles the 'retrieve' action, returning the cached value or None on miss."""
        if key in cache:
            logger.debug("Cache HIT for key: '%s'", key)
            return cache[key]
        else:
            logger.debug("Cache MISS for key: '%s'", key)
            return None

    def _handle_store(self, cache: Dict[Any, Any], key: Any, value: Any) -> Any:
        """Handles the 'store' action, storing the value and returning it."""
        cache[key] = value
        logger.info("Stored item in cache for key: '%s'", key)
        return value

    def _handle_invalidate(self, cache: Dict[Any, Any], key: Any) -> Optional[Any]:
        """Handles the 'invalidate' action, removing the key and returning it if found."""
        if key in cache:
            del cache[key]
            logger.info("Invalidated item from cache for key: '%s'", key)
            return key
        else:
            logger.debug("Attempted to invalidate non-existent key: '%s'", key)
            return None