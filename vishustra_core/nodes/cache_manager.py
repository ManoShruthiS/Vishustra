import logging
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManager(BaseNode):
    """
    A processing node designed for managing data caching operations within the Vishustra framework.

    This node interacts with a cache store provided via the execution context, supporting
    'get', 'set', and 'clear_all' operations. It allows for flexible integration of
    caching mechanisms into various parts of an LLM orchestration pipeline.

    The cache store is expected to be a mutable dictionary-like object found in the
    `context` under the key 'cache_store'. If not found, a basic in-memory dictionary
    is initialized for the current execution.

    Operations:
    - 'get': Attempts to retrieve a value associated with 'cache_key'. Returns the
             cached value on a hit, or `None` on a miss.
    - 'set': Stores the incoming `data` under the specified 'cache_key'. Returns the
             data that was just stored.
    - 'clear_all': Empties the entire cache store. Returns `None`.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes a cache operation based on the provided context.

        Args:
            data (Any): The input data. For 'set' operations, this is the value to be cached.
                        For 'get' and 'clear_all', this input is typically ignored.
            context (Dict[str, Any]): A dictionary containing shared execution context.
                                      Expected keys:
                                      - 'cache_store': The dictionary-like object to be used as the cache.
                                                       If not present, a new in-memory dictionary is
                                                       created and assigned to this key within the context.
                                      - 'cache_key' (str): The specific key for 'get' or 'set' operations.
                                                           This is a required parameter for these actions.
                                      - 'cache_action' (str): The desired cache operation ('get', 'set',
                                                              'clear_all'). Defaults to 'get' if not specified.

        Returns:
            Any: The cached value on a 'get' hit, the stored value on 'set', or `None` for
                 a 'get' miss or for 'clear_all' operations.

        Raises:
            ValueError: If 'cache_store' in the context is not a dictionary-like object,
                        if 'cache_key' is missing for 'get' or 'set' actions, or if an
                        unrecognized 'cache_action' is specified.
        """
        # Retrieve or initialize the cache store from the context
        cache_store = context.get('cache_store')
        if cache_store is None:
            logger.info(
                f"[{self.node_name}] No 'cache_store' found in context. Initializing a new in-memory cache for this pipeline."
            )
            cache_store = {}
            context['cache_store'] = cache_store  # Store it back in context for other nodes or subsequent calls
        elif not isinstance(cache_store, Dict):
            logger.error(
                f"[{self.node_name}] Invalid 'cache_store' type in context. Expected a dict-like object, got {type(cache_store).__name__}."
            )
            raise ValueError(
                f"Invalid 'cache_store' type. Expected a dict-like object, got {type(cache_store).__name__}."
            )

        cache_action = context.get('cache_action', 'get').lower()
        cache_key = context.get('cache_key')

        if cache_action == 'get':
            if cache_key is None:
                logger.error(f"[{self.node_name}] 'cache_key' is a mandatory parameter for the 'get' action.")
                raise ValueError("'cache_key' is required for 'get' action.")

            cached_value: Optional[Any] = cache_store.get(cache_key)
            if cached_value is not None:
                logger.debug(f"[{self.node_name}] Cache hit for key: '{cache_key}'.")
                return cached_value
            else:
                logger.debug(f"[{self.node_name}] Cache miss for key: '{cache_key}'.")
                return None

        elif cache_action == 'set':
            if cache_key is None:
                logger.error(f"[{self.node_name}] 'cache_key' is a mandatory parameter for the 'set' action.")
                raise ValueError("'cache_key' is required for 'set' action.")

            cache_store[cache_key] = data
            logger.debug(f"[{self.node_name}] Value successfully cached for key: '{cache_key}'.")
            return data

        elif cache_action == 'clear_all':
            cache_store.clear()
            logger.info(f"[{self.node_name}] Cache cleared successfully.")
            return None

        else:
            logger.error(f"[{self.node_name}] Encountered an unknown cache action: '{cache_action}'.")
            raise ValueError(
                f"Unknown cache action: '{cache_action}'. Expected 'get', 'set', or 'clear_all'."
            )