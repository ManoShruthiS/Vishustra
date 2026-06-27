import logging
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A Vishustra node designed to manage a key-value cache store.

    This node provides core cache operations:
    - 'get': Retrieves a value associated with a given key.
    - 'set': Stores or updates a value for a specific key.
    - 'delete': Removes a key-value pair from the cache.
    - 'clear': Empties the entire cache.

    The actual cache store (a mutable dictionary-like object) is expected
    to be supplied within the `context` dictionary under the key 'cache_store'.
    This design promotes stateless nodes and external cache management.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes a specified cache management operation.

        The `data` input must be a dictionary containing an 'action' key,
        which dictates the operation to perform. Depending on the action,
        additional 'key' and 'value' keys may be required within `data`.

        Expected `data` input formats:
        - To retrieve a value: `{'action': 'get', 'key': 'my_unique_key'}`
        - To store or update a value: `{'action': 'set', 'key': 'my_unique_key', 'value': my_data}`
        - To remove a key-value pair: `{'action': 'delete', 'key': 'my_unique_key'}`
        - To clear the entire cache: `{'action': 'clear'}`

        The `context` dictionary must contain a 'cache_store' key, whose value
        should be a mutable dictionary-like object to serve as the cache.

        Args:
            data: A dictionary specifying the cache operation and its parameters.
            context: A dictionary containing shared resources, including the 'cache_store'.

        Returns:
            The outcome of the cache operation:
            - For 'get': The cached value if found, otherwise `None`.
            - For 'set': `True` upon successful storage/update.
            - For 'delete': `True` if the key was deleted, `False` if the key was not found.
            - For 'clear': `True` upon successful cache clearing.

        Raises:
            TypeError: If the `data` input is not a dictionary.
            ValueError: If the 'action' key is missing or specifies an unknown operation,
                        or if required 'key'/'value' parameters are absent for an action.
            KeyError: If the 'cache_store' is missing from the `context` dictionary.
        """
        if not isinstance(data, dict):
            logger.error("Invalid input data type for %s. Expected a dictionary, got %s.", self.node_name, type(data).__name__)
            raise TypeError(f"Data for {self.node_name} must be a dictionary, got {type(data).__name__}.")

        action = data.get('action')
        if action is None:
            logger.error("Missing 'action' key in input data for %s: %s", self.node_name, data)
            raise ValueError(f"{self.node_name} requires an 'action' key in the input data.")

        # Validate and retrieve the cache store from context
        if 'cache_store' not in context:
            logger.critical("Context missing 'cache_store' for %s. This is a critical configuration error.", self.node_name)
            raise KeyError(f"The 'cache_store' key must be present in the context dictionary for {self.node_name}.")

        cache_store: Dict[str, Any] = context['cache_store']
        logger.debug("%s received action '%s'. Cache keys (before): %s", self.node_name, action, list(cache_store.keys()))

        try:
            if action == 'get':
                key = data.get('key')
                if key is None:
                    logger.error("Missing 'key' for 'get' action in data for %s: %s", self.node_name, data)
                    raise ValueError(f"{self.node_name} 'get' action requires a 'key'.")
                
                value = cache_store.get(key)
                if value is not None:
                    logger.debug("%s: Cache hit for key '%s'.", self.node_name, key)
                else:
                    logger.debug("%s: Cache miss for key '%s'.", self.node_name, key)
                return value
            
            elif action == 'set':
                key = data.get('key')
                value = data.get('value')
                if key is None or value is None:
                    logger.error("Missing 'key' or 'value' for 'set' action in data for %s: %s", self.node_name, data)
                    raise ValueError(f"{self.node_name} 'set' action requires both 'key' and 'value'.")
                
                cache_store[key] = value
                logger.info("%s: Set cache entry for key '%s'.", self.node_name, key)
                return True

            elif action == 'delete':
                key = data.get('key')
                if key is None:
                    logger.error("Missing 'key' for 'delete' action in data for %s: %s", self.node_name, data)
                    raise ValueError(f"{self.node_name} 'delete' action requires a 'key'.")
                
                if key in cache_store:
                    del cache_store[key]
                    logger.info("%s: Deleted cache entry for key '%s'.", self.node_name, key)
                    return True
                else:
                    logger.warning("%s: Attempted to delete non-existent key '%s'.", self.node_name, key)
                    return False

            elif action == 'clear':
                cache_store.clear()
                logger.info("%s: Cleared all entries from the cache.", self.node_name)
                return True

            else:
                logger.error("Unknown cache action '%s' requested in data for %s: %s", action, self.node_name, data)
                raise ValueError(f"Unknown cache action '{action}' for {self.node_name}.")

        except (TypeError, ValueError, KeyError) as e:
            # Re-raise specific, caught exceptions as they indicate predictable input issues
            raise e
        except Exception as e:
            # Catch any other unexpected errors during cache operations
            logger.exception("%s: An unexpected error occurred during cache operation '%s' with data %s.", self.node_name, action, data)
            raise RuntimeError(f"An unhandled error occurred in {self.node_name} for action '{action}': {e}") from e # Re-raise for pipeline integrity

        finally:
            logger.debug("%s: Cache keys (after '%s'): %s", self.node_name, action, list(cache_store.keys()))