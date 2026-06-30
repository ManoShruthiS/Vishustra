from vishustra_core.nodes.base_node import BaseNode
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A processing node that acts as an in-memory cache manager.
    It supports 'get', 'set', 'delete', and 'clear_all' operations on cached data.

    Operations are specified via the 'operation' key in the context dictionary.

    - 'get': Retrieves data associated with 'data' (key). Returns the value or None.
    - 'set': Stores 'context['value']' associated with 'data' (key). Returns the stored value.
    - 'delete': Removes the entry for 'data' (key). Returns True if deleted, False if not found.
    - 'clear_all': Clears all entries from the cache. Returns True.
    """

    _cache: Dict[Any, Any]

    def __init__(self, initial_cache: Optional[Dict[Any, Any]] = None):
        """
        Initializes the CacheManagerNode.

        Args:
            initial_cache (Optional[Dict[Any, Any]]): An optional dictionary
                                                       to pre-populate the cache.
        """
        self._cache = initial_cache if initial_cache is not None else {}
        logger.info(f"[{self.node_name}] Initialized with {len(self._cache)} pre-existing entries.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, performing cache operations based on the context.

        Args:
            data (Any): The primary input for the cache operation, typically the key.
            context (Dict[str, Any]): A dictionary containing operational parameters.
                Expected keys in context:
                - 'operation' (str): The cache operation to perform ('get', 'set', 'delete', 'clear_all').
                                     Defaults to 'get'.
                - 'value' (Any, for 'set' operation): The value to store in the cache.

        Returns:
            Any: The result of the cache operation:
                 - For 'get': The cached value or None if not found.
                 - For 'set': The value that was set.
                 - For 'delete': True if the key was deleted, False if not found.
                 - For 'clear_all': True.

        Raises:
            ValueError: If an invalid key type is provided, an unknown operation is requested,
                        or 'set' operation is called without a 'value' in context.
        """
        operation = context.get('operation', 'get').lower()

        if operation != 'clear_all':
            # Keys must be hashable for dictionary operations
            if not isinstance(data, (str, int, float, bool, tuple, frozenset)) and data is not None:
                logger.error(f"[{self.node_name}] Invalid cache key type: {type(data)}. Key must be hashable.")
                raise ValueError(f"Cache key must be hashable. Received type: {type(data)}")
            key = data
        else:
            key = None # Key is irrelevant for 'clear_all'

        if operation == 'get':
            if key in self._cache:
                logger.debug(f"[{self.node_name}] Cache hit for key: '{key}'")
                return self._cache[key]
            else:
                logger.info(f"[{self.node_name}] Cache miss for key: '{key}'")
                return None

        elif operation == 'set':
            value = context.get('value')
            if value is None:
                logger.error(f"[{self.node_name}] 'set' operation requires a 'value' in context for key: '{key}'")
                raise ValueError("Context missing 'value' for 'set' operation.")
            
            self._cache[key] = value
            logger.info(f"[{self.node_name}] Cache set for key: '{key}'")
            return value

        elif operation == 'delete':
            if key in self._cache:
                del self._cache[key]
                logger.info(f"[{self.node_name}] Cache deleted key: '{key}'")
                return True
            else:
                logger.warning(f"[{self.node_name}] Attempted to delete non-existent key: '{key}'")
                return False

        elif operation == 'clear_all':
            self._cache.clear()
            logger.info(f"[{self.node_name}] All cache entries cleared.")
            return True

        else:
            logger.error(f"[{self.node_name}] Unknown cache operation: '{operation}'")
            raise ValueError(f"Unknown cache operation: '{operation}'")

