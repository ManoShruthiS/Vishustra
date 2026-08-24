import logging
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A processing node responsible for managing data caching operations within Vishustra.

    This node facilitates intelligent retrieval, storage, and deletion of
    data from a configured cache store. It expects cache operations
    (get, set, delete) to be specified in the input `data` payload,
    and a compatible cache client to be provided in the `context`.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes a caching request based on the provided data and contextual information.

        The `data` payload is expected to specify the cache operation and details:
        - 'cache_key' (str): The unique identifier for the cache entry. (Required)
        - 'action' (str): The caching operation to perform.
                          Supported values: 'get', 'set', 'delete'. Defaults to 'get'. (Optional)
        - 'value' (Any): The data payload to store, if 'action' is 'set'. (Optional)
        - 'ttl' (int): Time-to-live in seconds for 'set' operations. (Optional)

        The `context` payload must contain the actual cache client instance:
        - 'cache_store' (object): An object that implements the necessary cache methods
                                   (e.g., `.get(key)`, `.set(key, value, ttl)`, `.delete(key)`).
                                   This is a critical dependency for the node's functionality.

        Returns:
            Dict[str, Any]: A dictionary encapsulating the outcome of the cache operation:
                            - For 'get': `{'status': 'hit'|'miss', 'value': Any|None}`.
                            - For 'set': `{'status': 'success'|'failure', 'key': str}`.
                            - For 'delete': `{'status': 'success'|'failure', 'key': str}`.

        Raises:
            ValueError: If 'cache_key' is absent from `data`, 'cache_store' is missing
                        from `context`, or an unsupported action is requested.
            RuntimeError: If an underlying cache operation encounters an unexpected failure
                          or the provided `cache_store` does not conform to the expected interface.
        """
        cache_key: Optional[str] = data.get('cache_key')
        action: str = data.get('action', 'get').lower()
        value: Any = data.get('value')
        ttl: Optional[int] = data.get('ttl')

        if not cache_key:
            logger.error("CacheManagerNode received data without 'cache_key'. Provided data: %s", data)
            raise ValueError("Missing required 'cache_key' in input data for CacheManagerNode.")

        cache_store = context.get('cache_store')
        if not cache_store:
            logger.error("CacheManagerNode received context without 'cache_store'. Context keys present: %s", context.keys())
            raise ValueError("Missing required 'cache_store' in context. Cannot perform any cache operations.")

        try:
            if action == 'get':
                result = cache_store.get(cache_key)
                if result is not None:
                    logger.debug("Cache hit for key '%s'.", cache_key)
                    return {"status": "hit", "value": result}
                else:
                    logger.debug("Cache miss for key '%s'.", cache_key)
                    return {"status": "miss", "value": None}
            elif action == 'set':
                if not hasattr(cache_store, 'set'):
                    raise AttributeError("Cache store object does not implement 'set' method.")
                cache_store.set(cache_key, value, ttl=ttl)
                logger.info("Successfully set cache key '%s'.", cache_key)
                return {"status": "success", "key": cache_key}
            elif action == 'delete':
                if not hasattr(cache_store, 'delete'):
                    raise AttributeError("Cache store object does not implement 'delete' method.")
                cache_store.delete(cache_key)
                logger.info("Successfully deleted cache key '%s'.", cache_key)
                return {"status": "success", "key": cache_key}
            else:
                logger.error("Unsupported cache action '%s' requested for key '%s'.", action, cache_key)
                raise ValueError(f"Unsupported action: '{action}'. Action must be 'get', 'set', or 'delete'.")
        except AttributeError as e:
            logger.exception("Cache store object is missing a required method for action '%s'. Error: %s", action, e)
            raise RuntimeError(
                f"Cache store interface error: The 'cache_store' object in context does not implement "
                f"the necessary method for action '{action}'. Details: {e}"
            ) from e
        except Exception as e:
            logger.exception(
                "An unexpected error occurred during cache operation '%s' for key '%s'. Error: %s",
                action, cache_key, e
            )
            raise RuntimeError(
                f"Failed to perform cache operation '{action}' for key '{cache_key}': {e}"
            ) from e