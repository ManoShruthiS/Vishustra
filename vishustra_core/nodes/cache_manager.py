import logging
from typing import Any, Dict, Optional, Callable, Hashable

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A processing node designed to manage data caching within Vishustra orchestration workflows.

    This node primarily serves as a cache reader. It attempts to retrieve data from a provided
    cache instance based on a derived or explicit cache key.

    Behavior of `process` method:
    - If a cache hit occurs (data is found in the cache), the node returns the cached data.
      The `context` dictionary will be updated with `context['cache_hit'] = True`.
    - If a cache miss occurs (data is not found in the cache), the node returns the original
      input `data`. This signals to downstream nodes that the data needs to be computed.
      The `context` dictionary will be updated with `context['cache_hit'] = False`.

    Configuration expected in the `context` dictionary for `process`:
    - `cache_instance`: **Required**. The actual cache object. This object is expected
      to have a `get(key)` method. Examples include `dict`, `functools.lru_cache`,
      or custom cache implementations.
    - `cache_key`: **Optional**. An explicit, hashable key to use for cache lookup.
      If provided, `data` is ignored for key generation.
    - `cache_key_transform_func`: **Optional**. A callable of type `Callable[[Any, Dict[str, Any]], Hashable]`.
      If `cache_key` is not provided, this function will be used to transform the input `data`
      and the current `context` into a hashable cache key.
    - If neither `cache_key` nor `cache_key_transform_func` is provided, `str(data)` will be
      used as the default cache key.

    Output updates to the `context` dictionary:
    - `cache_hit`: A `bool` indicating `True` for a cache hit or `False` for a cache miss.
    - `cache_key_used`: The specific hashable key that was used for the cache lookup.

    Error Handling:
    - Raises `ValueError` if `cache_instance` is missing or `cache_key` is not hashable.
    - Raises `AttributeError` if `cache_instance` lacks a `get` method.
    - Raises `RuntimeError` for unexpected failures during cache lookup.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "CacheManager"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Attempts to retrieve data from the configured cache.

        Args:
            data: The input data for the node. In case of a cache miss, this `data`
                  is returned, typically to be passed to a downstream computation node.
                  It also serves as the basis for cache key derivation if no explicit
                  key is provided.
            context: A dictionary containing runtime parameters, including the cache
                     instance and key configuration.

        Returns:
            The cached value if a hit is detected, or the original `data` if a miss occurs.
            The `context` dictionary will always be updated with cache status.

        Raises:
            ValueError: If critical configuration (`cache_instance`, `cache_key` type) is invalid.
            AttributeError: If `cache_instance` does not implement the expected interface (e.g., `get` method).
            RuntimeError: For any unexpected errors during the cache operation.
        """
        cache_instance = context.get('cache_instance')
        if not cache_instance:
            logger.error("CacheManagerNode: 'cache_instance' is required in the context.")
            raise ValueError("Missing 'cache_instance' in context for CacheManagerNode.")

        cache_key: Hashable
        if 'cache_key' in context:
            cache_key = context['cache_key']
            if not isinstance(cache_key, Hashable):
                logger.error(f"CacheManagerNode: Provided 'cache_key' in context is not hashable (type: {type(cache_key)}).")
                raise ValueError("The 'cache_key' provided in context must be hashable.")
        else:
            key_transform_func: Optional[Callable[[Any, Dict[str, Any]], Hashable]] = context.get('cache_key_transform_func')
            if key_transform_func:
                if not callable(key_transform_func):
                    logger.error(f"CacheManagerNode: 'cache_key_transform_func' is provided but is not a callable (type: {type(key_transform_func)}).")
                    raise ValueError("'cache_key_transform_func' must be a callable.")
                try:
                    cache_key = key_transform_func(data, context)
                    if not isinstance(cache_key, Hashable):
                        logger.error(f"CacheManagerNode: Transform function returned a non-hashable key (type: {type(cache_key)}).")
                        raise ValueError("The 'cache_key_transform_func' must return a hashable key.")
                except Exception as e:
                    logger.error(f"CacheManagerNode: Error transforming cache key with provided function: {e}", exc_info=True)
                    raise ValueError(f"Failed to transform cache key: {e}") from e
            else:
                try:
                    cache_key = str(data)  # Default key derivation
                except Exception as e:
                    logger.error(f"CacheManagerNode: Error converting input data to string for cache key: {e}", exc_info=True)
                    raise ValueError(f"Could not derive cache key from data: {e}") from e
                
        context['cache_key_used'] = cache_key
        logger.debug(f"CacheManagerNode: Attempting to retrieve from cache with key: '{cache_key}'")

        try:
            # We expect cache_instance to have a 'get' method.
            # Using .get() returns None for missing keys without raising an error.
            cached_value = cache_instance.get(cache_key)
            
            # Differentiate between a missing key and a key explicitly cached as None.
            # If the cache_instance supports `__contains__`, use it for a precise check.
            is_key_present = hasattr(cache_instance, '__contains__') and cache_key in cache_instance

            if cached_value is not None or is_key_present:
                context['cache_hit'] = True
                logger.debug(f"CacheManagerNode: Cache hit for key '{cache_key}'.")
                return cached_value
            else:
                # Key is not present, and get() returned None.
                context['cache_hit'] = False
                logger.debug(f"CacheManagerNode: Cache miss for key '{cache_key}'. Returning original data.")
                return data
        except AttributeError:
            logger.error(f"CacheManagerNode: 'cache_instance' object of type {type(cache_instance)} does not have a 'get' method.")
            raise AttributeError(f"Invalid cache_instance: object of type {type(cache_instance)} must have a 'get' method.")
        except Exception as e:
            logger.error(f"CacheManagerNode: An unexpected error occurred during cache lookup for key '{cache_key}': {e}", exc_info=True)
            raise RuntimeError(f"Cache lookup failed for key '{cache_key}': {e}") from e