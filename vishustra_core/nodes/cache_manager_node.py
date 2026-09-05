import logging
from typing import Any, Dict, Optional, Callable
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    Manages caching operations within the Vishustra pipeline.

    This node checks if a computed result for the given input data
    is available in the shared cache. If a cached value is found,
    it returns the cached value and signals a cache hit. If not,
    it passes the original data through, signaling a cache miss for
    downstream nodes to process and potentially store the result.

    Configuration Parameters for initialization:
    - `cache_store_key`: The key under which the shared cache dictionary
      is expected in the 'context'. Defaults to 'cache_store'.
    - `cache_hit_context_key`: The key to set in 'context' to indicate
      a cache hit (True) or miss (False). Defaults to 'cache_hit'.
    - `cache_key_generator`: An optional callable that takes the input
      'data' and returns a string or hashable object to be used as
      the cache key. If `None`, the input 'data' itself must be hashable
      and will be used as the key.
    - `return_on_cache_hit`: If `True`, the node returns the cached value
      directly on a hit, replacing the original `data`. If `False`,
      it always returns the original input `data`, but still sets
      `cache_hit_context_key` in the context. Defaults to `True`.
    """

    def __init__(
        self,
        cache_store_key: str = 'cache_store',
        cache_hit_context_key: str = 'cache_hit',
        cache_key_generator: Optional[Callable[[Any], Any]] = None,
        return_on_cache_hit: bool = True
    ):
        if not isinstance(cache_store_key, str) or not cache_store_key:
            raise ValueError("`cache_store_key` must be a non-empty string.")
        if not isinstance(cache_hit_context_key, str) or not cache_hit_context_key:
            raise ValueError("`cache_hit_context_key` must be a non-empty string.")
        if cache_key_generator is not None and not callable(cache_key_generator):
            raise TypeError("`cache_key_generator` must be a callable or None.")

        self._cache_store_key = cache_store_key
        self._cache_hit_context_key = cache_hit_context_key
        self._cache_key_generator = cache_key_generator
        self._return_on_cache_hit = return_on_cache_hit
        logger.debug(
            f"CacheManagerNode initialized with cache_store_key='{self._cache_store_key}', "
            f"cache_hit_context_key='{self._cache_hit_context_key}', "
            f"return_on_cache_hit={self._return_on_cache_hit}"
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "CacheManager"

    def _generate_cache_key(self, data: Any) -> Any:
        """
        Helper method to generate the cache key from the input data.
        Uses a custom generator if provided, otherwise attempts to use data directly.
        """
        if self._cache_key_generator:
            try:
                key = self._cache_key_generator(data)
                # Ensure the generated key is hashable for dictionary lookup
                hash(key)
                return key
            except TypeError as e:
                logger.error(
                    f"Custom cache_key_generator produced a non-hashable key for data type {type(data)}. Error: {e}",
                    exc_info=True
                )
                raise TypeError(f"Custom cache key generator failed to produce a hashable key: {e}")
            except Exception as e:
                logger.error(f"Error executing custom cache_key_generator: {e}", exc_info=True)
                raise ValueError(f"Failed to generate cache key: {e}")
        else:
            try:
                hash(data)  # Test if data is hashable
                return data
            except TypeError:
                logger.error(
                    f"Input data of type {type(data)} is not hashable and no custom cache_key_generator "
                    "was provided. Cannot use data directly as cache key."
                )
                raise TypeError(
                    "Input data not hashable and no cache_key_generator provided. "
                    "Consider providing a `cache_key_generator` callable."
                )

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to check and manage cache state.

        Args:
            data: The input data, which will be used (or from which a key will be derived)
                  to look up items in the cache.
            context: A dictionary containing shared pipeline state, expected to include
                     the cache store.

        Returns:
            The cached value if a hit and `return_on_cache_hit` is True.
            Otherwise, the original input `data` is returned.

        Raises:
            KeyError: If the configured `cache_store_key` is not found in the context.
            TypeError: If the cache store is not a dictionary or the generated cache key is not hashable.
            ValueError: If a custom cache key generator fails.
        """
        cache_store = context.get(self._cache_store_key)
        if cache_store is None:
            error_msg = (
                f"Cache store key '{self._cache_store_key}' not found in the processing context. "
                "Ensure a cache dictionary or object is provided in the context."
            )
            logger.error(error_msg)
            raise KeyError(error_msg)

        if not isinstance(cache_store, Dict): # Can be extended to support custom cache interfaces
            error_msg = (
                f"Expected '{self._cache_store_key}' in context to be a dictionary-like object, "
                f"but received type {type(cache_store)}. "
                "Current implementation requires a Dict for the cache store."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        try:
            cache_key = self._generate_cache_key(data)
        except (TypeError, ValueError) as e:
            logger.error(
                f"Failed to generate cache key for input data of type {type(data)}. "
                f"Processing halted for CacheManagerNode. Error: {e}", exc_info=True
            )
            raise # Re-raise to propagate the error up the pipeline

        if cache_key in cache_store:
            cached_value = cache_store[cache_key]
            context[self._cache_hit_context_key] = True
            logger.info(f"Cache HIT for key: {cache_key}. Node '{self.node_name}' detected cached value.")
            if self._return_on_cache_hit:
                return cached_value
            else:
                return data # Still return original data, useful for observability or specialized flows
        else:
            context[self._cache_hit_context_key] = False
            logger.info(f"Cache MISS for key: {cache_key}. Node '{self.node_name}' passing original data through.")
            return data