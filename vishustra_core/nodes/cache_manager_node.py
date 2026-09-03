import logging
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A processing node responsible for managing data within a shared cache.

    This node supports 'get', 'set', and 'delete' operations on a cache
    instance provided in the context.

    Expected `context` keys:
    - 'cache_instance': The actual cache object (e.g., a dictionary, an LRU cache,
                        or an interface to an external caching system). This is
                        a mandatory component.
    - 'cache_operation': A string indicating the desired operation: 'get', 'set', or 'delete'.
                         Defaults to 'get' if not specified.
    - 'cache_value' (optional): Required for 'set' operations; the value to store.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "CacheManagerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Performs cache operations based on the provided data and context.

        Args:
            data (Any): The key for the cache operation.
            context (Dict[str, Any]): A dictionary containing execution context,
                                      including the 'cache_instance' and 'cache_operation'.

        Returns:
            Any: The result of the cache operation:
                 - For 'get': The cached value if found, otherwise None.
                 - For 'set': The value that was stored.
                 - For 'delete': True if the key was deleted, False if not found.

        Raises:
            ValueError: If 'cache_instance' is missing from context, an invalid
                        'cache_operation' is specified, or 'cache_value' is
                        missing for a 'set' operation.
            TypeError: If the 'cache_instance' does not support the required operations.
        """
        cache_instance: Optional[Any] = context.get('cache_instance')
        cache_operation: str = context.get('cache_operation', 'get').lower()
        
        if cache_instance is None:
            logger.error(f"[{self.node_name}] Missing 'cache_instance' in context for cache operation.")
            raise ValueError("Cache instance not provided in context.")

        key = data
        result = None

        try:
            if cache_operation == 'get':
                result = cache_instance.get(key)
                if result is None:
                    logger.debug(f"[{self.node_name}] Cache miss for key: '{key}'.")
                else:
                    logger.debug(f"[{self.node_name}] Cache hit for key: '{key}'.")
            elif cache_operation == 'set':
                value = context.get('cache_value')
                if value is None:
                    logger.error(f"[{self.node_name}] 'cache_value' is required in context for 'set' operation for key: '{key}'.")
                    raise ValueError(f"Missing 'cache_value' for set operation on key '{key}'.")
                
                cache_instance[key] = value # Assuming dictionary-like set behavior
                logger.info(f"[{self.node_name}] Set cache key: '{key}' with value: '{value}'.")
                result = value
            elif cache_operation == 'delete':
                if key in cache_instance:
                    del cache_instance[key] # Assuming dictionary-like delete behavior
                    logger.info(f"[{self.node_name}] Deleted cache key: '{key}'.")
                    result = True
                else:
                    logger.debug(f"[{self.node_name}] Key '{key}' not found for deletion.")
                    result = False
            else:
                logger.error(f"[{self.node_name}] Invalid cache operation specified: '{cache_operation}'.")
                raise ValueError(f"Unsupported cache operation: '{cache_operation}'.")

        except AttributeError as e:
            logger.error(f"[{self.node_name}] Cache instance does not support required operation for key '{key}'. Error: {e}")
            raise TypeError(f"Cache instance type does not support operation '{cache_operation}': {e}") from e
        except Exception as e:
            logger.error(f"[{self.node_name}] An unexpected error occurred during cache operation '{cache_operation}' for key '{key}'. Error: {e}")
            raise

        return result
