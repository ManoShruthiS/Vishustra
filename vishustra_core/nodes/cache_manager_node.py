import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for this module
logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A Vishustra node responsible for managing a simple key-value cache
    within the orchestration context.

    This node supports common cache operations like 'get', 'set', and 'delete',
    allowing other nodes in a pipeline to store and retrieve ephemeral data.
    The cache itself is stored in the `context` dictionary under the key
    `'cache_store'`.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this cache manager node.
        """
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes cache operations based on the input `data`.

        The `data` input is expected to be a dictionary specifying the operation
        and relevant details:
        -   `"operation"` (str): Must be "get", "set", or "delete".
        -   `"key"` (Any): The identifier for the cache entry.
        -   `"value"` (Any, optional): Required for "set" operation; the value
            to store in the cache.
        -   `"default"` (Any, optional): Used for "get" operation; the value
            to return if the key is not found in the cache. Defaults to `None`.

        The cache state is managed within the `context` dictionary at
        `context['cache_store']`. If this key is not present or not a dictionary,
        it will be initialized to an empty dictionary.

        Args:
            data (Any): A dictionary containing the cache operation instructions.
                        Example: `{"operation": "set", "key": "user_id", "value": 123}`
                        Example: `{"operation": "get", "key": "user_id", "default": 0}`
            context (Dict[str, Any]): The current orchestration context, used
                                      to store and access the cache.

        Returns:
            Any: The result of the cache operation:
                 - For "get": The cached value, or the default value if not found.
                 - For "set": The value that was successfully set.
                 - For "delete": The key that was deleted, or `None` if not found.

        Raises:
            ValueError: If the input `data` is malformed, missing required keys,
                        or specifies an unknown operation.
            RuntimeError: For unexpected errors during cache manipulation.
        """
        if not isinstance(data, dict):
            logger.error(
                "CacheManagerNode received malformed data. Expected a dictionary, got %s.",
                type(data).__name__
            )
            raise ValueError("Invalid input data format. Expected a dictionary.")

        operation = data.get("operation")
        key = data.get("key")

        if not operation:
            logger.error("CacheManagerNode received data without 'operation': %s", data)
            raise ValueError("Missing 'operation' in input data.")
        if key is None:
            logger.error(
                "CacheManagerNode received data without 'key' for operation '%s': %s",
                operation, data
            )
            raise ValueError("Missing 'key' in input data for cache operation.")

        # Ensure the cache store exists and is a dictionary in the context
        if 'cache_store' not in context or not isinstance(context['cache_store'], dict):
            logger.warning(
                "Cache store not found or is not a dictionary in context. Initializing an empty 'cache_store'."
            )
            context['cache_store'] = {}

        cache_store = context['cache_store']
        result = None

        try:
            if operation == "get":
                result = cache_store.get(key, data.get("default", None))
                if key in cache_store:
                    logger.debug("Cache hit for key '%s'. Value retrieved.", key)
                else:
                    logger.debug(
                        "Cache miss for key '%s'. Returning default value: %s.",
                        key, result
                    )
            elif operation == "set":
                value = data.get("value")
                cache_store[key] = value
                result = value
                logger.debug("Cache set operation: key='%s', value set.", key)
            elif operation == "delete":
                if key in cache_store:
                    del cache_store[key]
                    result = key
                    logger.debug("Cache delete operation: key='%s' removed.", key)
                else:
                    logger.debug(
                        "Cache delete operation: key='%s' not found, no action taken.",
                        key
                    )
                    result = None # Indicate that the key was not present to delete
            else:
                logger.error("Invalid cache operation '%s' received.", operation)
                raise ValueError(
                    f"Invalid cache operation: '{operation}'. Must be 'get', 'set', or 'delete'."
                )
        except Exception as e:
            logger.exception(
                "An unexpected error occurred during cache operation '%s' for key '%s'.",
                operation, key
            )
            raise RuntimeError(
                f"Failed to perform cache operation '{operation}' for key '{key}': {e}"
            ) from e

        return result