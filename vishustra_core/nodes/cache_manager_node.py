import logging
import time
from typing import Any, Dict, Optional, Tuple

# Assuming BaseNode is located here as per project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    Manages a transient in-memory cache within the Vishustra processing context.

    This node provides functionality to interact with a shared context-based cache,
    allowing other nodes to store, retrieve, delete, and clear data. Cache entries
    can optionally be configured with a Time-To-Live (TTL) in seconds.

    The actual cache data structure is stored and retrieved from the provided
    `context` dictionary under a specified namespace, enabling multiple distinct
    caches within a single orchestration flow.
    """

    def __init__(self, cache_namespace: str = "default_cache"):
        """
        Initializes the CacheManagerNode.

        Args:
            cache_namespace: The key under which this node will store and retrieve
                             its managed cache within the processing context. This
                             allows for multiple CacheManagerNode instances to
                             operate on different, isolated caches within the same
                             context. Must be a non-empty string.
        
        Raises:
            ValueError: If `cache_namespace` is not a non-empty string.
        """
        if not isinstance(cache_namespace, str) or not cache_namespace.strip():
            logger.error(f"Invalid cache_namespace provided: '{cache_namespace}'. Must be a non-empty string.")
            raise ValueError("cache_namespace must be a non-empty string.")
        
        self._cache_namespace = cache_namespace.strip()
        logger.debug(f"CacheManagerNode initialized with namespace: '{self._cache_namespace}'")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node, including its cache namespace.
        """
        return f"CacheManagerNode::{self._cache_namespace}"

    def _get_or_initialize_cache(self, context: Dict[str, Any]) -> Dict[str, Tuple[Any, Optional[float]]]:
        """
        Retrieves the cache dictionary from the context based on its namespace.
        If the cache does not exist in the context, it initializes an empty one.

        Args:
            context: The shared context dictionary for the current orchestration run.

        Returns:
            The dictionary representing the cache for this node's namespace.
        """
        if self._cache_namespace not in context:
            context[self._cache_namespace] = {}
            logger.info(f"Initialized new cache dictionary for namespace '{self._cache_namespace}' in context.")
        return context[self._cache_namespace]

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes cache management requests based on the input data.

        The `data` input is expected to be a dictionary specifying the cache
        operation and its parameters.

        Expected `data` formats for different actions:
        - Get an entry: `{'action': 'get', 'key': 'my_key'}`
        - Set an entry: `{'action': 'set', 'key': 'my_key', 'value': 'my_value', 'ttl': 3600}`
          (TTL is optional, in seconds. A non-positive TTL implies no expiration.)
        - Delete an entry: `{'action': 'delete', 'key': 'my_key'}`
        - Clear all entries: `{'action': 'clear'}`

        Args:
            data: A dictionary containing the cache operation details.
            context: The shared context dictionary where the cache is managed.

        Returns:
            A dictionary containing the result of the cache operation.
            - For 'get': `{'action': 'get', 'key': ..., 'status': 'hit'|'miss', 'value': ..., 'expired': True|False}`
            - For 'set': `{'action': 'set', 'key': ..., 'status': 'success', 'ttl_seconds': ...}`
            - For 'delete': `{'action': 'delete', 'key': ..., 'status': 'success'|'not_found'}`
            - For 'clear': `{'action': 'clear', 'status': 'success', 'cleared_entries': ...}`
            - For errors: `{'status': 'error', 'message': ...}`
        
        Raises:
            ValueError: If the input `data` is not a dictionary.
        """
        if not isinstance(data, dict):
            logger.error(f"Invalid data type for CacheManagerNode. Expected dict, got {type(data)}.")
            raise ValueError("Input 'data' for CacheManagerNode must be a dictionary.")

        action = data.get('action')
        if not action or not isinstance(action, str):
            logger.warning(f"Missing or invalid 'action' key in input data: {data}")
            return {"status": "error", "message": "Missing or invalid 'action' specified in data."}

        action = action.lower().strip()
        cache = self._get_or_initialize_cache(context)
        result: Dict[str, Any] = {"action": action, "node_name": self.node_name}

        try:
            if action == 'get':
                key = data.get('key')
                if key is None:
                    logger.warning(f"Attempted 'get' action without 'key' in data: {data}")
                    return {"status": "error", "message": "Key is required for 'get' action."}
                
                result['key'] = key
                entry = cache.get(key)
                
                if entry is None:
                    result.update({"status": "miss", "value": None, "expired": False})
                    logger.debug(f"Cache miss for key '{key}' in namespace '{self._cache_namespace}'.")
                else:
                    value, expiry = entry
                    if expiry is not None and time.time() > expiry:
                        del cache[key] # Remove expired entry
                        result.update({"status": "miss", "value": None, "expired": True})
                        logger.debug(f"Cache entry for key '{key}' in namespace '{self._cache_namespace}' was expired and removed.")
                    else:
                        result.update({"status": "hit", "value": value, "expired": False})
                        logger.debug(f"Cache hit for key '{key}' in namespace '{self._cache_namespace}'.")

            elif action == 'set':
                key = data.get('key')
                value = data.get('value')
                ttl = data.get('ttl') # Time-To-Live in seconds

                if key is None or value is None:
                    logger.warning(f"Attempted 'set' action without 'key' or 'value' in data: {data}")
                    return {"status": "error", "message": "Key and value are required for 'set' action."}

                expiry: Optional[float] = None
                if ttl is not None:
                    try:
                        ttl_float = float(ttl)
                        if ttl_float > 0:
                            expiry = time.time() + ttl_float
                        else:
                            logger.debug(f"Non-positive TTL ({ttl_float}) provided for key '{key}'. Entry will not expire.")
                            ttl = None # Store as None to indicate no expiration
                    except (TypeError, ValueError):
                        logger.warning(f"Invalid TTL value '{ttl}' for key '{key}'. Ignoring TTL and setting entry without expiration.")
                        ttl = None # Ensure ttl in result reflects actual outcome
                
                cache[key] = (value, expiry)
                result.update({"status": "success", "key": key, "ttl_seconds": ttl})
                logger.debug(f"Set cache entry for key '{key}' in namespace '{self._cache_namespace}' with TTL: {ttl if ttl is not None else 'N/A'}s.")

            elif action == 'delete':
                key = data.get('key')
                if key is None:
                    logger.warning(f"Attempted 'delete' action without 'key' in data: {data}")
                    return {"status": "error", "message": "Key is required for 'delete' action."}
                
                if key in cache:
                    del cache[key]
                    result.update({"status": "success", "key": key})
                    logger.debug(f"Deleted cache entry for key '{key}' in namespace '{self._cache_namespace}'.")
                else:
                    result.update({"status": "not_found", "key": key})
                    logger.debug(f"Attempted to delete non-existent key '{key}' in namespace '{self._cache_namespace}'.")

            elif action == 'clear':
                initial_size = len(cache)
                cache.clear()
                result.update({"status": "success", "cleared_entries": initial_size})
                logger.info(f"Cleared all {initial_size} entries from cache in namespace '{self._cache_namespace}'.")

            else:
                logger.warning(f"Received unknown cache action '{action}' in data: {data}")
                return {"status": "error", "message": f"Unknown action: '{action}'."}
        
        except Exception as e:
            logger.exception(f"An unexpected error occurred during cache operation '{action}' for namespace '{self._cache_namespace}': {e}")
            result = {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}
        
        return result