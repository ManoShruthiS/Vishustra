import logging
from typing import Any, Dict, Optional, MutableMapping

# Assuming BaseNode is available in the specified path
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A Vishustra node designed to manage caching operations within a pipeline.

    This node provides functionality to interact with a cache instance
    passed via the processing context. It supports operations such as
    retrieving data from the cache ('get'), storing data into the cache ('set'),
    and invalidating existing cache entries ('invalidate').

    The node acts as a transparent pass-through for the primary data payload,
    returning either the cached value (on a 'get' hit) or the original input
    data (on 'set', 'invalidate', 'get' miss, or error conditions).
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by performing a specified cache operation.

        The `context` dictionary is crucial for controlling this node's behavior
        and is expected to contain:
        - 'cache_instance': A mutable mapping (e.g., a dict or a custom cache object)
                            that supports dict-like operations (`__getitem__`,
                            `__setitem__`, `__delitem__`, `get`, `__contains__`).
                            If not provided or not a mutable mapping, the node
                            will default to a 'noop' operation.
        - 'cache_operation': A string ('get', 'set', 'invalidate', 'noop')
                             indicating the desired cache interaction.
                             Defaults to 'noop' if not specified.
        - 'cache_key': The key (any hashable type) to be used for the cache operation.
                       Required for 'get', 'set', and 'invalidate' operations.

        Args:
            data: The primary data payload flowing through the pipeline.
                  For 'set' operations, this data is stored in the cache.
                  For 'get' operations, this data is returned on a cache miss.
            context: A dictionary containing operational parameters for the node,
                     including the cache instance, operation type, and key.

        Returns:
            The result of the cache operation:
            - Cached value if 'get' operation is a hit.
            - Original `data` if 'set' or 'invalidate' operation.
            - Original `data` if 'get' operation is a miss.
            - Original `data` if no valid cache operation is performed or an error occurs.
        """
        cache_instance_raw: Any = context.get('cache_instance')
        operation: str = context.get('cache_operation', 'noop').lower()
        key: Any = context.get('cache_key')

        # Validate cache_instance: it must be a mutable mapping
        if not isinstance(cache_instance_raw, MutableMapping):
            logger.warning(
                "No valid 'cache_instance' (expected MutableMapping) found in context for "
                "CacheManagerNode. Performing no-op and passing data through."
            )
            return data

        cache_instance: MutableMapping[Any, Any] = cache_instance_raw

        if operation not in ['get', 'set', 'invalidate', 'noop']:
            logger.error(
                f"Invalid cache operation '{operation}' specified in context for "
                "CacheManagerNode. Performing no-op and passing data through."
            )
            return data

        if operation != 'noop' and key is None:
            logger.error(
                f"Missing 'cache_key' in context for operation '{operation}' in "
                "CacheManagerNode. Performing no-op and passing data through."
            )
            return data

        try:
            if operation == 'get':
                if key in cache_instance:
                    cached_value = cache_instance[key]
                    logger.debug(f"Cache hit for key '{key}'. Returning cached value.")
                    return cached_value
                else:
                    logger.debug(f"Cache miss for key '{key}'. Returning original data.")
                    return data
            elif operation == 'set':
                cache_instance[key] = data
                logger.debug(f"Data set in cache for key '{key}'.")
                return data
            elif operation == 'invalidate':
                if key in cache_instance:
                    del cache_instance[key]
                    logger.debug(f"Cache entry for key '{key}' invalidated.")
                else:
                    logger.debug(f"Attempted to invalidate non-existent key '{key}'.")
                return data
            elif operation == 'noop':
                logger.debug("CacheManagerNode performing no-op as specified.")
                return data
            # This 'else' branch should theoretically not be reached due to prior validation
            else:
                logger.warning(f"Unhandled cache operation '{operation}'. Returning original data.")
                return data
        except TypeError as te: # e.g., unhashable key
            logger.error(
                f"TypeError during cache operation '{operation}' with key '{key}' in "
                f"CacheManagerNode: {te}. Returning original data."
            )
            return data
        except Exception as e:
            # Catch any other unexpected errors during cache interaction
            logger.exception(
                f"An unexpected error occurred during cache operation '{operation}' for key '{key}' "
                f"in CacheManagerNode: {e}. Returning original data."
            )
            return data