import logging
from typing import Any, Dict, Optional

# Assuming vishustra_core is installed and base_node is accessible
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A Vishustra node that provides in-memory caching capabilities.
    It supports 'set', 'get', 'delete', and 'clear' operations on cached data.
    
    This node maintains its own internal dictionary as an in-memory cache.
    
    Context parameters for process method:
    - 'operation' (str): Required. Specifies the cache operation ('set', 'get', 'delete', 'clear').
    - 'key' (str): Required for 'set', 'get', 'delete' operations. The key to store/retrieve/delete data.
    - 'default_value' (Any): Optional, only for 'get' operation. The value to return if the key is not found
                             in the cache, and no explicit `data` is provided to the node's process method.
    
    Data parameter for process method:
    - For 'set' operation: The value to be cached.
    - For 'get' operation: Can serve as a fallback value if the key is not found in the cache and
                           'default_value' is not provided in the context.
    - For 'delete', 'clear' operations: This parameter is typically ignored.
    """

    def __init__(self):
        """Initializes the CacheManagerNode with an empty in-memory cache."""
        self._cache: Dict[str, Any] = {}
        logger.debug(f"{self.node_name} initialized with an empty cache.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "CacheManagerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes cache operations based on the provided context.

        Args:
            data (Any): The primary input data for the node. Its interpretation depends on the operation:
                        - For 'set' operation: This `data` is the value to be stored in the cache.
                        - For 'get' operation: This `data` can act as a fallback return value if the
                          requested `key` is not found in the cache and `context['default_value']` is also absent.
                        - For 'delete' or 'clear' operations: This `data` parameter is ignored.
            context (Dict[str, Any]): A dictionary containing operation details.
                                       - 'operation' (str): Required. Must be one of 'set', 'get', 'delete', 'clear'.
                                       - 'key' (str): Required for 'set', 'get', 'delete'. The identifier for the cache entry.
                                       - 'default_value' (Any): Optional. For 'get' operation, this value is returned
                                         if the key is not found and takes precedence over the `data` parameter.

        Returns:
            Any: The result of the cache operation:
                 - 'set': The value that was successfully stored in the cache.
                 - 'get': The cached value; if not found, returns `context['default_value']`,
                          then the `data` parameter, and finally `None` if all fallbacks are absent.
                 - 'delete': The value that was removed from the cache; `None` if the key was not found.
                 - 'clear': True if the cache was successfully cleared.
        
        Raises:
            ValueError: If 'operation' is missing or invalid, or if 'key' is missing or invalid for
                        operations that require it ('set', 'get', 'delete').
            RuntimeError: For any unexpected errors encountered during the cache operation.
        """
        operation: Optional[str] = context.get('operation')
        key: Optional[str] = context.get('key')
        
        if not operation:
            logger.error(f"{self.node_name}: Missing 'operation' in context for processing.")
            raise ValueError("Context must contain an 'operation' key ('set', 'get', 'delete', 'clear').")

        if operation not in ['set', 'get', 'delete', 'clear']:
            logger.error(f"{self.node_name}: Invalid operation '{operation}' specified in context.")
            raise ValueError(f"Invalid operation '{operation}'. Must be one of 'set', 'get', 'delete', 'clear'.")

        # Validate 'key' for operations that explicitly require it
        if operation in ['set', 'get', 'delete']:
            if not isinstance(key, str) or not key:
                logger.error(f"{self.node_name}: Missing or invalid 'key' for operation '{operation}'. Key must be a non-empty string.")
                raise ValueError(f"Context must contain a non-empty string 'key' for '{operation}' operation.")
        
        try:
            if operation == 'set':
                self._cache[key] = data
                logger.info(f"{self.node_name}: Successfully set key '{key}'.")
                return data
            
            elif operation == 'get':
                cached_value = self._cache.get(key)
                if cached_value is not None:
                    logger.debug(f"{self.node_name}: Retrieved key '{key}' from cache.")
                    return cached_value
                
                # Fallback hierarchy: context['default_value'] -> data parameter -> None
                default_value_from_context = context.get('default_value')
                if default_value_from_context is not None:
                    logger.debug(f"{self.node_name}: Key '{key}' not found, returning 'default_value' from context.")
                    return default_value_from_context
                
                if data is not None:
                    logger.debug(f"{self.node_name}: Key '{key}' not found, returning `data` parameter as fallback.")
                    return data

                logger.info(f"{self.node_name}: Key '{key}' not found, returning None as no fallback was provided.")
                return None
                
            elif operation == 'delete':
                if key in self._cache:
                    deleted_value = self._cache.pop(key)
                    logger.info(f"{self.node_name}: Successfully deleted key '{key}'.")
                    return deleted_value
                else:
                    logger.warning(f"{self.node_name}: Attempted to delete non-existent key '{key}'. No action taken.")
                    return None
                    
            elif operation == 'clear':
                self._cache.clear()
                logger.info(f"{self.node_name}: Cache successfully cleared.")
                return True

        except Exception as e:
            # Catch broader exceptions for robustness, log details, and re-raise as RuntimeError
            logger.exception(
                f"{self.node_name}: An unexpected error occurred during '{operation}' operation with key "
                f"'{key if key else 'N/A'}'. Error: {e}"
            )
            raise RuntimeError(f"Cache operation '{operation}' failed: {e}") from e