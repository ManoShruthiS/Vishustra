import logging
from typing import Any, Dict, Optional, Tuple

# Assuming BaseNode is located here based on project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A Vishustra processing node that acts as an in-memory cache manager.

    This node supports common cache operations:
    - Retrieving a value (get)
    - Storing a value (set)
    - Deleting a value (delete)

    Operations are determined by the 'cache_operation' key in the context.
    The 'data' parameter to the process method is always interpreted as the cache key.
    """

    def __init__(self):
        """
        Initializes the CacheManagerNode with an empty in-memory cache.
        """
        self._cache: Dict[Any, Any] = {}  # Stores Key -> Value
        logger.debug("CacheManagerNode initialized with an empty in-memory cache.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "CacheManager"

    def _get_from_cache(self, key: Any) -> Tuple[Optional[Any], bool]:
        """
        Internal method to retrieve a value from the cache.

        Args:
            key (Any): The key associated with the value to retrieve.

        Returns:
            Tuple[Optional[Any], bool]: A tuple containing the value if found
                                        (or None if not found) and a boolean
                                        indicating whether a cache hit occurred.
        """
        if key in self._cache:
            value = self._cache[key]
            logger.debug(f"Cache hit for key: '{key}'")
            return value, True
        logger.debug(f"Cache miss for key: '{key}'")
        return None, False

    def _set_in_cache(self, key: Any, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Internal method to store a value in the cache.

        Args:
            key (Any): The key to associate with the value.
            value (Any): The value to store.
            ttl (Optional[int]): Time-to-live in seconds. (Reserved for future implementation).

        Returns:
            bool: True if the value was successfully set, False otherwise.
        """
        try:
            self._cache[key] = value
            logger.info(f"Cache set for key: '{key}' (TTL: {ttl if ttl is not None else 'None'})")
            # Future expansion: Implement actual TTL management here.
            return True
        except Exception as e:
            logger.error(f"Error setting cache for key '{key}': {e}", exc_info=True)
            return False

    def _delete_from_cache(self, key: Any) -> Tuple[bool, str]:
        """
        Internal method to delete a value from the cache.

        Args:
            key (Any): The key of the item to delete.

        Returns:
            Tuple[bool, str]: A tuple where the first element is True if deleted,
                              False otherwise. The second element is a descriptive
                              message ("Deleted", "Not Found", or an error message).
        """
        try:
            if key in self._cache:
                del self._cache[key]
                logger.info(f"Cache deleted for key: '{key}'")
                return True, "Deleted"
            logger.debug(f"Attempted to delete non-existent key from cache: '{key}'")
            return False, "Not Found"
        except Exception as e:
            logger.error(f"Error deleting cache for key '{key}': {e}", exc_info=True)
            return False, str(e)

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a cache operation based on the provided key (`data`) and `context`.

        Args:
            data (Any): The cache key for the operation. Must be a hashable type.
            context (Dict[str, Any]): A dictionary containing operation details.
                                      Expected keys:
                                      - "cache_operation" (str, optional): "get", "set", "delete". Defaults to "get".
                                      - "cache_value" (Any, optional): The value to store for "set" operation.
                                      - "ttl" (int, optional): Time-to-live in seconds for "set" operation.
                                                               Currently reserved for future implementation.

        Returns:
            Dict[str, Any]: A dictionary indicating the result of the operation.
                            Possible statuses:
                            - "hit": Item found in cache. Contains "key" and "value".
                            - "miss": Item not found in cache. Contains "key".
                            - "set": Item successfully stored. Contains "key" and "value".
                            - "deleted": Item successfully deleted. Contains "key".
                            - "key_not_found": Attempted delete on a non-existent key. Contains "key" and "message".
                            - "error": An error occurred during the operation. Contains "message" and optionally "key".
        """
        operation = context.get("cache_operation", "get").lower()
        
        # Basic validation for cache key type
        if not isinstance(data, (str, int, float, bool, tuple)):
            # More complex objects can be keys if hashable, but restrict to common immutable types
            # for a robust initial implementation. Custom objects would require __hash__ and __eq__.
            logger.warning(
                f"Invalid data type used as cache key: {type(data)}. Key must be hashable and immutable."
            )
            return {"status": "error", "message": f"Invalid cache key type: {type(data)}."}

        try:
            if operation == "get":
                value, hit = self._get_from_cache(data)
                if hit:
                    return {"status": "hit", "key": data, "value": value}
                else:
                    return {"status": "miss", "key": data}
            
            elif operation == "set":
                value_to_set = context.get("cache_value")
                ttl = context.get("ttl") # Future use for actual TTL implementation
                if value_to_set is None:
                    logger.warning(
                        f"Attempted 'set' operation for key '{data}' without 'cache_value' in context."
                    )
                    return {"status": "error", "message": "Missing 'cache_value' for 'set' operation."}
                
                if self._set_in_cache(data, value_to_set, ttl):
                    return {"status": "set", "key": data, "value": value_to_set}
                else:
                    return {"status": "error", "message": f"Failed to set cache for key: '{data}'"}
            
            elif operation == "delete":
                success, msg = self._delete_from_cache(data)
                if success:
                    return {"status": "deleted", "key": data}
                elif msg == "Not Found":
                    return {"status": "key_not_found", "key": data, "message": "Key not found for deletion."}
                else:
                    return {"status": "error", "key": data, "message": f"Failed to delete cache: {msg}"}
            
            else:
                logger.warning(f"Unknown cache operation '{operation}' for key: '{data}'")
                return {"status": "error", "message": f"Unknown cache operation: '{operation}'"}
        
        except Exception as e:
            logger.error(
                f"Unhandled exception during cache operation '{operation}' for key '{data}': {e}", 
                exc_info=True
            )
            return {"status": "error", "message": f"An unexpected error occurred: {e}"}

