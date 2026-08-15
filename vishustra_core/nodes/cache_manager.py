import logging
from typing import Any, Dict, Optional

# Assuming BaseNode is available at this path within the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManager(BaseNode):
    """
    A Vishustra node designed to manage a simple in-memory key-value cache.

    This node facilitates common caching operations such as 'get', 'set', and
    'invalidate' based on parameters provided in the processing context.
    It's suitable for scenarios where transient data needs to be stored and
    retrieved quickly within an orchestration flow.
    """

    def __init__(self):
        """
        Initializes the CacheManager node, setting up an empty dictionary
        to serve as the in-memory cache storage.
        """
        self._cache: Dict[str, Any] = {}
        logger.debug("CacheManager node initialized with an empty in-memory cache.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes a cache operation ('get', 'set', or 'invalidate') based on
        the 'cache_operation' and 'cache_key' specified in the context.

        Args:
            data (Any): The input data. For 'set' operations, this is the
                        value to be cached. For 'get' and 'invalidate',
                        it's passed through but not directly used by the cache logic.
            context (Dict[str, Any]): A dictionary containing parameters for
                                     the caching operation. Expected keys:
                                     - 'cache_operation' (str): The desired action
                                                                ('get', 'set', 'invalidate').
                                     - 'cache_key' (str): The unique identifier
                                                          for the cache entry.

        Returns:
            Any:
                - For 'get': The value retrieved from the cache if a hit, otherwise `None`.
                - For 'set': The `data` that was just stored in the cache.
                - For 'invalidate': `None`.

        Raises:
            ValueError: If 'cache_operation' or 'cache_key' are missing from
                        the context or are of an invalid type, or if an
                        unsupported 'cache_operation' is provided.
        """
        cache_operation: Optional[str] = context.get("cache_operation")
        cache_key: Optional[str] = context.get("cache_key")

        if not isinstance(cache_operation, str):
            logger.error("Validation failed: 'cache_operation' missing or not a string in context.")
            raise ValueError("Context requires a 'cache_operation' (str) to perform caching.")

        if not isinstance(cache_key, str):
            logger.error("Validation failed: 'cache_key' missing or not a string in context for operation '%s'.", cache_operation)
            raise ValueError("Context requires a 'cache_key' (str) to perform caching.")

        if cache_operation == "get":
            return self._get_from_cache(cache_key)
        elif cache_operation == "set":
            return self._set_in_cache(cache_key, data)
        elif cache_operation == "invalidate":
            self._invalidate_cache_entry(cache_key)
            return None  # Invalidation operations typically do not return data
        else:
            logger.error("Unsupported cache operation '%s' specified for key '%s'.", cache_operation, cache_key)
            raise ValueError(
                f"Unsupported cache_operation: '{cache_operation}'. "
                "Supported operations are 'get', 'set', 'invalidate'."
            )

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """
        Helper method to retrieve a value from the internal cache.
        """
        try:
            value = self._cache[key]
            logger.info("Cache HIT for key '%s'.", key)
            return value
        except KeyError:
            logger.debug("Cache MISS for key '%s'.", key)
            return None
        except Exception as e:
            logger.exception("An unexpected error occurred while retrieving key '%s' from cache.", key)
            return None

    def _set_in_cache(self, key: str, value: Any) -> Any:
        """
        Helper method to store a value in the internal cache.
        """
        self._cache[key] = value
        logger.info("Cache SET for key '%s'. Data type: %s.", key, type(value).__name__)
        return value

    def _invalidate_cache_entry(self, key: str) -> None:
        """
        Helper method to remove an entry from the internal cache.
        """
        try:
            del self._cache[key]
            logger.info("Cache INVALIDATED for key '%s'.", key)
        except KeyError:
            logger.warning("Attempted to invalidate non-existent cache key '%s'. No action taken.", key)
        except Exception as e:
            logger.exception("An unexpected error occurred while invalidating key '%s' from cache.", key)