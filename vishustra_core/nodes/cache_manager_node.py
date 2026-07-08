import logging
import hashlib
import json
from typing import Any, Dict, Optional

# Assuming the specified import path for BaseNode
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A processing node responsible for managing data caching within the Vishustra pipeline.

    This node implements a caching mechanism primarily for read operations.
    When `process` is called:
    1. It generates a stable cache key based on the input `data`.
    2. It attempts to retrieve a previously computed result using this key from
       the 'cache_store' provided in the `context`.
    
    Behavior based on cache status:
    - If a cache hit occurs: The node immediately returns the cached result,
      effectively short-circuiting any subsequent nodes in the pipeline for this request.
      It sets `context['cache_hit']` to `True`.
    - If a cache miss occurs: The node passes the original `data` along for further
      processing by downstream nodes. It sets `context['cache_hit']` to `False`
      and stores the generated cache key in `context['cache_key_for_storage']`.
      This key can then be used by an orchestrator or a subsequent dedicated node
      to store the final result of the downstream processing if desired.

    The 'cache_store' is expected to be a dictionary-like object (e.g., `dict`,
    or an object implementing `collections.abc.MutableMapping`) provided
    within the `context` dictionary under the key `'cache_store'`.

    Example `context` setup:
    `context = {'cache_store': my_in_memory_cache_dict}`
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "CacheManager"

    def _generate_cache_key(self, data: Any) -> str:
        """
        Generates a stable and consistent cache key from the input data.

        This method prioritizes JSON serialization for complex data structures
        (like dictionaries and lists) to ensure consistent key generation regardless
        of insertion order. For primitive types, it uses their string representation.
        The generated string is then hashed using SHA256 to produce a fixed-size key.

        Args:
            data: The input data for which to generate a cache key.

        Returns:
            A SHA256 hexadecimal string representing the cache key.

        Raises:
            ValueError: If the input data is not suitable for cache key generation
                        (e.g., not JSON serializable and not a primitive type).
        """
        if isinstance(data, (str, int, float, bool, type(None))):
            key_str = str(data)
        else:
            try:
                # Attempt to serialize to JSON for complex structures (dicts, lists)
                # sort_keys ensures consistent key order, ensure_ascii=False handles Unicode
                key_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
            except TypeError as e:
                # If data is not JSON serializable and not a primitive,
                # we cannot reliably generate a consistent key.
                error_msg = (
                    f"Data of type '{type(data).__name__}' is not suitable for cache key generation. "
                    f"It must be JSON serializable or a primitive type. Original error: {e}"
                )
                logger.error(error_msg, exc_info=True)
                raise ValueError(error_msg) from e
            
        return hashlib.sha256(key_str.encode('utf-8')).hexdigest()

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by attempting to retrieve it from the cache.

        Args:
            data: The input data to be processed (or checked in cache).
            context: A dictionary containing shared pipeline context, expected
                     to include 'cache_store' (a dict-like object where cache
                     entries are stored).

        Returns:
            The cached result if a cache hit occurs, otherwise the original `data`
            is returned for further processing by downstream nodes.

        Raises:
            ValueError: If 'cache_store' is not found or is not a dictionary-like
                        object in the context, or if the input data fails key generation.
        """
        cache_store = context.get('cache_store')
        # We check for Dict, but ideally it could be any MutableMapping for flexibility.
        # For simplicity and common use cases, Dict is a good starting point.
        if not isinstance(cache_store, Dict):
            error_msg = (
                "CacheManagerNode requires a 'cache_store' (dict-like object) "
                "in the context to operate."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            cache_key = self._generate_cache_key(data)
        except ValueError as e:
            # If key generation fails, we cannot interact with the cache.
            # Log the error, treat as a cache miss, and pass the original data.
            # No cache_key_for_storage is set, as a valid key couldn't be formed.
            logger.warning(f"Skipping cache check due to failed key generation. Error: {e}")
            context['cache_hit'] = False
            return data

        # Store the generated cache key in context for potential future writes
        # by an orchestrator or another node if a miss occurs.
        context['cache_key_for_storage'] = cache_key

        if cache_key in cache_store:
            cached_value = cache_store[cache_key]
            context['cache_hit'] = True
            logger.debug(f"Cache hit for key '{cache_key[:8]}...'. Returning cached value.")
            return cached_value
        else:
            context['cache_hit'] = False
            logger.debug(f"Cache miss for key '{cache_key[:8]}...'. Passing data for processing.")
            return data
