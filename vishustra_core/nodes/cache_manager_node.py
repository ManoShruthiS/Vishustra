import logging
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A Vishustra processing node that manages a data cache within the execution context.

    This node supports common caching operations such as getting, setting,
    deleting, and clearing cache entries. It expects operation details in the
    'data' input dictionary and interacts with a cache store managed within
    the 'context' dictionary.

    The cache store is typically a simple Python dictionary, allowing for
    in-memory caching across nodes sharing the same context. It can be
    identified by a configurable key within the context.
    """

    # Default key under which the cache store will be managed in the context
    _DEFAULT_CACHE_CONTEXT_KEY = "vishustra_global_cache_store"

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by performing a specified cache operation.

        The 'data' input must be a dictionary containing at least an "operation" key.
        Depending on the operation, other keys like "key", "value", and "cache_key"
        may be required.

        The 'context' dictionary is used to hold the actual cache store. If a cache
        store identified by `cache_context_key` (defaulting to `_DEFAULT_CACHE_CONTEXT_KEY`)
        is not found or is not a dictionary, an empty dictionary will be initialized
        in the context for subsequent operations within the same pipeline run.

        Args:
            data (Any): A dictionary specifying the cache operation and its parameters.
                        Expected keys:
                        - "operation" (str): The desired cache action ("get", "set", "delete", "clear").
                        - "key" (str, optional): The cache key to operate on (required for "get", "set", "delete").
                        - "value" (Any, optional): The value to store (required for "set").
                        - "cache_key" (str, optional): The specific key in the context where the
                                                       cache dictionary is stored. Defaults to
                                                       `_DEFAULT_CACHE_CONTEXT_KEY`.

            context (Dict[str, Any]): The shared execution context dictionary,
                                       where the cache store is managed.

        Returns:
            Any: The result of the cache operation:
                - For "get": The cached value if found, otherwise `None`.
                - For "set": The value that was successfully stored.
                - For "delete": The value that was deleted, or `None` if the key did not exist.
                - For "clear": A confirmation string (e.g., "Cache cleared").
                - On error: Raises an exception, or `None` if the error is handled internally
                            without re-raising.

        Raises:
            ValueError: If 'data' is not a dictionary, or if required parameters
                        (like 'operation', 'key', 'value') are missing or invalid
                        for the specified operation.
            Exception: For unexpected errors during cache operations, to propagate
                       failure upstream.
        """
        if not isinstance(data, dict):
            logger.error(f"[{self.node_name}] Invalid input data type: expected a dictionary, got {type(data)}.")
            raise ValueError(f"Input data for '{self.node_name}' must be a dictionary.")

        operation: Optional[str] = data.get("operation")
        cache_context_key: str = data.get("cache_key", self._DEFAULT_CACHE_CONTEXT_KEY)

        # Retrieve or initialize the cache store from the context
        cache_store = context.get(cache_context_key)
        if not isinstance(cache_store, dict):
            logger.warning(
                f"[{self.node_name}] Cache store not found or invalid type under context key "
                f"'{cache_context_key}'. Initializing an empty dictionary cache for this context run."
            )
            cache_store = {}
            context[cache_context_key] = cache_store  # Store the new cache dict back into context

        key: Optional[str] = data.get("key")
        value: Any = data.get("value")
        result: Any = None

        try:
            if operation == "get":
                if key is None:
                    logger.error(f"[{self.node_name}] 'get' operation requires a 'key'.")
                    raise ValueError("'get' operation requires a 'key'.")
                result = cache_store.get(key)
                if result is not None:
                    logger.debug(f"[{self.node_name}] Cache hit for key '{key}'.")
                else:
                    logger.debug(f"[{self.node_name}] Cache miss for key '{key}'.")
                return result

            elif operation == "set":
                if key is None:
                    logger.error(f"[{self.node_name}] 'set' operation requires a 'key'.")
                    raise ValueError("'set' operation requires a 'key'.")
                # 'value' can be None, so we only check if it's missing from the data dict
                if "value" not in data:
                    logger.error(f"[{self.node_name}] 'set' operation requires a 'value'.")
                    raise ValueError("'set' operation requires a 'value'.")
                
                cache_store[key] = value
                logger.debug(f"[{self.node_name}] Cache set for key '{key}'.")
                return value

            elif operation == "delete":
                if key is None:
                    logger.error(f"[{self.node_name}] 'delete' operation requires a 'key'.")
                    raise ValueError("'delete' operation requires a 'key'.")
                if key in cache_store:
                    deleted_value = cache_store.pop(key)
                    logger.debug(f"[{self.node_name}] Cache deleted for key '{key}'.")
                    return deleted_value
                else:
                    logger.warning(f"[{self.node_name}] Attempted to delete non-existent key '{key}'.")
                    return None

            elif operation == "clear":
                cache_store.clear()
                logger.debug(f"[{self.node_name}] Cache cleared successfully under context key '{cache_context_key}'.")
                return "Cache cleared"

            else:
                logger.error(f"[{self.node_name}] Unknown or unsupported cache operation: '{operation}'.")
                raise ValueError(f"Unknown cache operation: '{operation}'")

        except ValueError as ve:
            # Re-raise explicit ValueErrors for malformed input
            raise ve
        except Exception as e:
            # Catch all other unexpected errors during cache interaction
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during operation "
                f"'{operation}' on cache key '{key}' in context key '{cache_context_key}': {e}",
                exc_info=True
            )
            raise  # Re-raise to signal a failure in the pipeline