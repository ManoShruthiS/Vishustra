import logging
from typing import Any, Dict, Optional
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A specialized node for managing transient state and caching intermediate results
    within the Vishustra orchestration pipeline. 
    
    This node facilitates 'get', 'set', and 'evict' operations to optimize 
    LLM API calls and expensive data transformations.
    """

    def __init__(self):
        # In a production environment, this would interface with Redis or a similar 
        # distributed store. For current modularity, we use a controlled internal registry.
        self._store: Dict[str, Any] = {}

    @property
    def node_name(self) -> str:
        """Returns the canonical name of the caching node."""
        return "CacheManagerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes cache operations based on directives provided in the context.
        
        Args:
            data: The payload to be cached or the default return value on a miss.
            context: Dictionary containing operational metadata:
                - 'cache_action': Literal['get', 'set', 'evict']
                - 'cache_key': The unique identifier for the cached object.
        
        Returns:
            The retrieved data on 'get' (if hit), otherwise returns the input 'data'.
        """
        action = context.get("cache_action")
        key = context.get("cache_key")

        if not action or not key:
            logger.debug("CacheManagerNode: Missing 'cache_action' or 'cache_key' in context. Passing data through.")
            return data

        try:
            if action == "get":
                return self._retrieve(key, data)
            elif action == "set":
                return self._store_data(key, data)
            elif action == "evict":
                return self._evict(key, data)
            else:
                logger.warning(f"CacheManagerNode: Received unrecognized action '{action}'.")
                return data
        except KeyError as ke:
            logger.error(f"CacheManagerNode: Key error during '{action}' operation: {ke}")
            return data
        except Exception as e:
            logger.exception(f"CacheManagerNode: Unexpected failure during cache {action}: {str(e)}")
            return data

    def _retrieve(self, key: str, fallback: Any) -> Any:
        """Handles cache lookup logic."""
        if key in self._store:
            logger.info(f"Cache Hit: {key}")
            return self._store[key]
        
        logger.info(f"Cache Miss: {key}")
        return fallback

    def _store_data(self, key: str, value: Any) -> Any:
        """Handles cache persistence logic."""
        if value is None:
            logger.warning(f"CacheManagerNode: Attempted to cache NoneType for key '{key}'. Operation aborted.")
            return value

        self._store[key] = value
        logger.info(f"Cache Set: {key}")
        return value

    def _evict(self, key: str, data: Any) -> Any:
        """Handles cache invalidation logic."""
        if key in self._store:
            del self._store[key]
            logger.info(f"Cache Evicted: {key}")
        else:
            logger.debug(f"Cache Evict: Key {key} not found in store.")
        return data

    def clear_all(self) -> None:
        """Flushes the entire node cache registry."""
        self._store.clear()
        logger.info("CacheManagerNode: Internal store cleared.")