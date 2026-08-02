import logging
import json
from hashlib import md5
from typing import Any, Dict, Optional, Union

# Assuming vishustra_core.nodes.base_node exists in the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A Vishustra node designed for intelligent caching operations within the
    orchestration flow. It facilitates getting, setting, and deleting data
    from a shared cache store provided in the context.

    This node is crucial for optimizing performance by avoiding redundant
    computations (e.g., LLM calls) when results are already available.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "CacheManagerNode"

    def _generate_cache_key(self, data: Any) -> Optional[str]:
        """
        Generates a deterministic cache key from the input data.
        It attempts to hash simple types directly or serialize complex types
        to JSON before hashing to ensure a stable and unique key.
        """
        try:
            # Handle directly hashable types
            if isinstance(data, (str, int, float, bool, bytes)):
                return str(data)
            # For more complex types, serialize to JSON and then hash
            # sort_keys ensures consistent key order for dicts
            # default=str handles non-serializable objects by converting them to string
            serialized_data = json.dumps(data, sort_keys=True, default=str)
            return md5(serialized_data.encode('utf-8')).hexdigest()
        except TypeError as e:
            logger.warning(
                f"CacheManagerNode: Could not generate a cache key from input data "
                f"directly or via JSON serialization. Data type: {type(data)}. Error: {e}"
            )
            return None
        except Exception as e:
            logger.error(f"CacheManagerNode: An unexpected error occurred while generating a cache key: {e}")
            return None

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes caching operations based on the configuration provided in the context.

        The `context` dictionary is expected to contain:
        - 'cache_store': A mutable dictionary-like object that serves as the cache
                         (e.g., an in-memory dict, or a wrapper for Redis/Memcached).
                         This is a mandatory parameter.
        - 'cache_key': (Optional[str]) An explicit string to use as the cache key.
                       If not provided, a key will be generated from the `data` argument.
        - 'cache_action': (str) The desired cache operation: 'get', 'set', or 'delete'.
                          Defaults to 'get' if not specified.
        - 'value_to_cache': (Any) Used primarily with the 'set' action. This is the
                            specific value to store in the cache. If not provided for
                            a 'set' action, the `data` argument itself will be stored.

        Outputs for 'get' action:
        - If a cache hit occurs: The retrieved cached value is returned.
        - If a cache miss occurs: The original `data` is returned.
        In both 'get' scenarios, `context['cache_hit']` will be set to `True` or `False`.

        Outputs for 'set' or 'delete' actions:
        - The original `data` argument is passed through this node, as these actions
          are primarily for their side effects on the cache.
        """
        cache_store = context.get("cache_store")
        if not isinstance(cache_store, dict):
            logger.error(
                f"{self.node_name}: A 'cache_store' (expected dict-like) was not found "
                f"or was of an invalid type ({type(cache_store)}) in the context. "
                f"Cannot perform cache operations. Passing data through."
            )
            context["cache_hit"] = False  # Ensure flag is always set, even on error
            return data

        explicit_key = context.get("cache_key")
        cache_key = explicit_key if explicit_key is not None else self._generate_cache_key(data)

        if cache_key is None:
            logger.error(
                f"{self.node_name}: Failed to determine a valid cache key from explicit "
                f"context key or input data. Cannot perform cache operations. "
                f"Passing data through."
            )
            context["cache_hit"] = False
            return data

        cache_action = context.get("cache_action", "get").lower()

        if cache_action == "get":
            return self._handle_get(cache_store, cache_key, data, context)
        elif cache_action == "set":
            return self._handle_set(cache_store, cache_key, data, context)
        elif cache_action == "delete":
            return self._handle_delete(cache_store, cache_key, data, context)
        else:
            logger.warning(
                f"{self.node_name}: Received an unknown 'cache_action': '{cache_action}'. "
                f"Defaulting to 'get' operation."
            )
            return self._handle_get(cache_store, cache_key, data, context)

    def _handle_get(self, cache_store: Dict[str, Any], cache_key: str, data: Any, context: Dict[str, Any]) -> Any:
        """Handles the 'get' cache action, retrieving data if available."""
        try:
            if cache_key in cache_store:
                cached_value = cache_store[cache_key]
                context["cache_hit"] = True
                logger.debug(f"{self.node_name}: Cache hit for key '{cache_key}'.")
                return cached_value
            else:
                context["cache_hit"] = False
                logger.debug(f"{self.node_name}: Cache miss for key '{cache_key}'.")
                return data
        except Exception as e:
            logger.error(f"{self.node_name}: Error during 'get' operation for key '{cache_key}': {e}")
            context["cache_hit"] = False
            return data

    def _handle_set(self, cache_store: Dict[str, Any], cache_key: str, data: Any, context: Dict[str, Any]) -> Any:
        """Handles the 'set' cache action, storing data into the cache."""
        try:
            value_to_store = context.get("value_to_cache", data)
            cache_store[cache_key] = value_to_store
            logger.info(f"{self.node_name}: Cache entry set for key '{cache_key}'.")
            # The 'set' operation is primarily for its side effect; pass original data through.
            return data
        except Exception as e:
            logger.error(f"{self.node_name}: Error during 'set' operation for key '{cache_key}': {e}")
            return data

    def _handle_delete(self, cache_store: Dict[str, Any], cache_key: str, data: Any, context: Dict[str, Any]) -> Any:
        """Handles the 'delete' cache action, removing an entry from the cache."""
        try:
            if cache_key in cache_store:
                del cache_store[cache_key]
                logger.info(f"{self.node_name}: Cache entry deleted for key '{cache_key}'.")
            else:
                logger.debug(f"{self.node_name}: Attempted to delete non-existent cache key '{cache_key}'.")
            # The 'delete' operation is primarily for its side effect; pass original data through.
            return data
        except Exception as e:
            logger.error(f"{self.node_name}: Error during 'delete' operation for key '{cache_key}': {e}")
            return data
