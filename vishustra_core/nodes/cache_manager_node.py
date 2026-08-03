import logging
from typing import Any, Dict, Optional, MutableMapping

# Assume BaseNode is available in the specified path
# In a real project, this would be a relative import if within the same package
# or a full package import if BaseNode is part of an installed library.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheMiss:
    """
    A sentinel object to indicate a cache miss.
    This allows distinguishing between a cached None value and a truly absent value.
    """
    def __repr__(self) -> str:
        return "<CacheMiss>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CacheMiss)

    def __hash__(self) -> int:
        return hash(self.__class__)

# Singleton instance for convenience
CACHE_MISS = CacheMiss()

class CacheManagerNode(BaseNode):
    """
    A processing node responsible for interacting with a shared cache store.

    This node can perform two primary operations:
    1. 'get': Attempts to retrieve a value from the cache using the input `data` as the key.
             Returns the cached value on hit, or a `CacheMiss` sentinel on miss.
    2. 'set': Stores a value into the cache. The input `data` is expected to be a
             dictionary `{'key': Any, 'value': Any}`. It returns the stored value.

    The cache store itself is expected to be provided in the `context` dictionary
    under the key 'cache_store', and must be a mutable mapping (e.g., a dict).
    """

    def __init__(self, operation: str = 'get'):
        """
        Initializes the CacheManagerNode with a specified operation.

        Args:
            operation (str): The cache operation to perform. Must be 'get' or 'set'.
        
        Raises:
            ValueError: If an unsupported operation is provided.
        """
        if operation not in ['get', 'set']:
            raise ValueError(f"Invalid operation '{operation}'. Must be 'get' or 'set'.")
        self._operation = operation
        logger.debug(f"CacheManagerNode initialized with operation: '{self._operation}'")

    @property
    def node_name(self) -> str:
        """Returns the name of the node, including its operation."""
        return f"CacheManagerNode_{self._operation.capitalize()}"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data based on the node's configured operation.

        Args:
            data (Any):
                - If operation is 'get': The key to look up in the cache.
                - If operation is 'set': A dictionary `{'key': Any, 'value': Any}`
                                         containing the key and value to store.
            context (Dict[str, Any]): The shared context dictionary, expected to
                                      contain 'cache_store' (a mutable mapping).

        Returns:
            Any:
                - If operation is 'get' and cache hit: The cached value.
                - If operation is 'get' and cache miss: `CACHE_MISS` sentinel.
                - If operation is 'set': The value that was stored.

        Raises:
            RuntimeError: If 'cache_store' is missing from context or not a mutable mapping.
            TypeError: If input `data` does not conform to the expected type for the operation.
            KeyError: If 'key' or 'value' are missing from `data` for 'set' operation.
        """
        cache_store: Optional[MutableMapping[Any, Any]] = context.get('cache_store')

        if not isinstance(cache_store, MutableMapping):
            error_msg = f"Context missing 'cache_store' or it's not a mutable mapping. Found: {type(cache_store)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        if self._operation == 'get':
            key_to_get = data
            if not isinstance(key_to_get, (str, int, float, tuple)):
                # Consider what types can be used as cache keys; often hashable types.
                # For simplicity, restrict to common immutable types or allow Any if cache handles it.
                logger.warning(f"Attempting to retrieve cache entry with non-primitive key type: {type(key_to_get)}. Key: {key_to_get}")

            try:
                cached_value = cache_store.get(key_to_get, CACHE_MISS)
                if cached_value is CACHE_MISS:
                    logger.info(f"Cache MISS for key: {key_to_get}")
                else:
                    logger.info(f"Cache HIT for key: {key_to_get}")
                return cached_value
            except Exception as e:
                logger.error(f"Error accessing cache for key '{key_to_get}': {e}", exc_info=True)
                # Depending on policy, might re-raise, or treat as a miss
                return CACHE_MISS

        elif self._operation == 'set':
            if not isinstance(data, dict):
                error_msg = f"For 'set' operation, expected data to be a dict {{'key': ..., 'value': ...}}, but got {type(data)}."
                logger.error(error_msg)
                raise TypeError(error_msg)

            try:
                key_to_set = data['key']
                value_to_set = data['value']
            except KeyError as e:
                error_msg = f"For 'set' operation, data dictionary must contain '{e.args[0]}' key. Got: {data.keys()}"
                logger.error(error_msg)
                raise KeyError(error_msg)

            try:
                cache_store[key_to_set] = value_to_set
                logger.info(f"Cache SET: Stored value for key: {key_to_set}")
                return value_to_set  # Return the value stored for potential further processing
            except Exception as e:
                logger.error(f"Error setting cache for key '{key_to_set}': {e}", exc_info=True)
                raise RuntimeError(f"Failed to set cache entry for key '{key_to_set}': {e}") from e

        # This part should theoretically be unreachable due to constructor validation
        # but added for robustness.
        else:
            error_msg = f"Internal error: Unsupported operation '{self._operation}' in process method."
            logger.critical(error_msg)
            raise RuntimeError(error_msg)
