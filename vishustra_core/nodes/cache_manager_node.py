import logging
from typing import Any, Dict, Optional

# Assuming BaseNode is located here as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A Vishustra processing node responsible for managing data caching operations.

    This node supports 'get', 'set', and 'delete' actions on a cache store
    provided within the processing context. It's designed to either retrieve
    data from cache, store data into cache, or remove data from cache based
    on the provided `action` and `cache_key`.

    The actual cache store (e.g., an in-memory dictionary, a connection to Redis,
    etc.) is expected to be available in the `context` dictionary under the key
    'cache_store'. For simulation purposes, this node treats 'cache_store' as a
    mutable dictionary.
    """

    def __init__(self):
        """
        Initializes the CacheManagerNode.
        """
        logger.debug("CacheManagerNode initialized.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "CacheManagerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes data related to caching operations.

        This method expects the `data` input to be a dictionary specifying
        the caching `action`, the `cache_key`, and optionally the `value`
        for 'set' operations. The `context` dictionary must provide the
        'cache_store' where operations will be performed.

        Args:
            data (Any): A dictionary containing:
                        - 'action' (str): The desired caching operation ('get', 'set', 'delete').
                        - 'cache_key' (Any): The key for the cache entry.
                        - 'value' (Any, optional): The value to store for the 'set' action.
            context (Dict[str, Any]): The processing context, which *must* include
                                       'cache_store' (expected to be a mutable dictionary
                                       or a cache-like object supporting `__contains__`,
                                       `__getitem__`, `__setitem__`, `__delitem__`).

        Returns:
            Any:
                - For 'get' action: The cached value if found, otherwise `None` (cache miss).
                - For 'set' action: The value that was successfully stored.
                - For 'delete' action: `True` if the key was found and deleted, `False` otherwise.

        Raises:
            TypeError: If `data` is not a dictionary, or if 'cache_store' in `context`
                       is not a dictionary-like object.
            ValueError: If 'cache_store' is missing from `context`, or if required
                        keys ('action', 'cache_key', or 'value' for 'set') are missing
                        or invalid in `data`.
        """
        if not isinstance(data, dict):
            logger.error("Invalid input data for CacheManagerNode: expected a dictionary.")
            raise TypeError(f"Input `data` for {self.node_name} must be a dictionary, got {type(data).__name__}.")

        action: Optional[str] = data.get('action')
        cache_key: Any = data.get('cache_key')
        value: Any = data.get('value')

        if 'cache_store' not in context:
            logger.error("Missing 'cache_store' in context for CacheManagerNode.")
            raise ValueError(f"{self.node_name} requires 'cache_store' in the processing context.")

        cache_store = context['cache_store']
        # While typically a dict, allowing any object with dict-like interface for flexibility
        if not hasattr(cache_store, '__contains__') or \
           not hasattr(cache_store, '__getitem__') or \
           not hasattr(cache_store, '__setitem__') or \
           not hasattr(cache_store, '__delitem__'):
            logger.error(f"Invalid 'cache_store' type in context: expected a dict-like object, got {type(cache_store)}.")
            raise TypeError(f"'cache_store' in context must be a dictionary-like object, got {type(cache_store).__name__}.")

        if not action or not isinstance(action, str):
            logger.error("Missing or invalid 'action' in data for CacheManagerNode.")
            raise ValueError(f"`data` must contain a string 'action' ('get', 'set', 'delete') for {self.node_name}.")
        
        if cache_key is None:
            # While None could theoretically be a key, it's often an indicator of missing data
            logger.error("Missing 'cache_key' in data for CacheManagerNode.")
            raise ValueError(f"`data` must contain 'cache_key' for {self.node_name}.")

        action = action.lower()

        if action == 'get':
            if cache_key in cache_store:
                cached_value = cache_store[cache_key]
                logger.debug(f"Cache hit for key: '{cache_key}' by {self.node_name}.")
                return cached_value
            else:
                logger.debug(f"Cache miss for key: '{cache_key}' by {self.node_name}.")
                return None
        elif action == 'set':
            if 'value' not in data:
                logger.error(f"Missing 'value' in data for 'set' action for key '{cache_key}'.")
                raise ValueError(f"`data` must contain 'value' for 'set' action in {self.node_name}.")
            
            cache_store[cache_key] = value
            logger.debug(f"Stored data for key: '{cache_key}' by {self.node_name}.")
            return value
        elif action == 'delete':
            if cache_key in cache_store:
                del cache_store[cache_key]
                logger.debug(f"Deleted data for key: '{cache_key}' by {self.node_name}.")
                return True
            else:
                logger.debug(f"Attempted to delete non-existent key: '{cache_key}' by {self.node_name}.")
                return False
        else:
            logger.error(f"Unsupported action '{action}' for CacheManagerNode.")
            raise ValueError(f"Unsupported action: '{action}'. Must be 'get', 'set', or 'delete' for {self.node_name}.")