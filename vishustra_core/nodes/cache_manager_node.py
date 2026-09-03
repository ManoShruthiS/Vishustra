import logging
import json
from typing import Any, Dict, Callable, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A Vishustra processing node that manages a data cache.

    This node primarily serves two functions within a Vishustra pipeline:
    1.  **Cache Read (via `process` method):**
        -   When `process` is called, it attempts to retrieve a value from the cache
            using a key derived from the input `data`.
        -   If a **cache hit** occurs, the node returns the cached value directly.
            This action effectively short-circuits subsequent expensive computations
            in the pipeline for this specific data path. The `context` is updated
            with `context['cache_hit'] = True`.
        -   If a **cache miss** occurs, the node passes the original `data` through
            to allow subsequent nodes to perform the necessary computation. The `context`
            is updated with `context['cache_hit'] = False` and the derived
            `cache_key` is stored in `context['cache_key']` for potential later use.

    2.  **Cache Write (via `store` method):**
        -   The `store` method can be explicitly invoked (e.g., by the orchestration
            framework or another "writer" node) to save a computed result into the cache.
            This is typically done after a cache miss has led to a successful computation.

    The cache itself can be an internal dictionary managed by the node, or an
    externally provided dictionary, enabling shared caching across multiple nodes.
    A custom key extraction function can be supplied, otherwise a robust default
    extractor is used to handle various data types.
    """

    _node_name: str = "CacheManager"

    def __init__(
        self,
        cache_store: Optional[Dict[Any, Any]] = None,
        cache_key_extractor: Optional[Callable[[Any], Any]] = None,
    ):
        """
        Initializes the CacheManagerNode.

        Args:
            cache_store (Optional[Dict[Any, Any]]): An optional dictionary to use as
                the cache backing store. If `None`, an internal dictionary will be
                initialized and used. This allows for sharing a cache instance across
                multiple `CacheManagerNode` instances or for external cache management.
            cache_key_extractor (Optional[Callable[[Any], Any]]): A callable that
                takes the input `data` (Any) and returns a hashable key (Any) suitable
                for caching. If `None`, a robust default extractor will be used, which
                handles strings, numbers, and consistently serializes dictionaries.
        """
        self._cache_store: Dict[Any, Any] = cache_store if cache_store is not None else {}

        if cache_key_extractor is None:
            self._cache_key_extractor: Callable[[Any], Any] = self._default_key_extractor
        elif not callable(cache_key_extractor):
            logger.error(
                f"{self.node_name}: Provided 'cache_key_extractor' is not a callable. "
                "Falling back to default key extraction mechanism."
            )
            self._cache_key_extractor = self._default_key_extractor
        else:
            self._cache_key_extractor = cache_key_extractor

        logger.debug(f"{self.node_name} initialized. Using cache_store type: {type(self._cache_store)}")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return self._node_name

    def _default_key_extractor(self, data: Any) -> Any:
        """
        A robust default method to extract a cache key from input data.
        It handles common data types for consistent key generation.
        """
        if isinstance(data, (str, int, float, bool)):
            return data
        if isinstance(data, dict):
            try:
                # Sort dictionary keys to ensure a consistent JSON string representation
                # regardless of insertion order, which is crucial for cache key consistency.
                return json.dumps(data, sort_keys=True)
            except TypeError:
                logger.warning(
                    f"{self.node_name}: Could not JSON-serialize dictionary data for cache key. "
                    f"Falling back to 'repr()'. Data type: {type(data)}"
                )
                return repr(data)
        # Fallback for other complex types or objects that are not JSON serializable.
        try:
            return repr(data)
        except Exception as e:
            logger.error(
                f"{self.node_name}: Failed to generate cache key using 'repr()' for data type "
                f"{type(data)}: {e}. Using 'str()' as a final fallback, which may not guarantee uniqueness."
            )
            return str(data)

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by attempting to retrieve it from the cache.

        If a cache hit, the cached value is returned. If a cache miss, the original
        data is returned, and context is updated to indicate the miss.

        Args:
            data (Any): The input data that needs to be processed or retrieved from cache.
            context (Dict[str, Any]): A mutable dictionary for sharing state, metadata,
                                      and flags across nodes in the pipeline.
                                      This method will update `context['cache_hit']` (bool)
                                      and `context['cache_key']` (Any).

        Returns:
            Any: The cached value if a hit; otherwise, the original `data` is returned
                 to allow subsequent pipeline nodes to compute the result.
        """
        cache_key = None
        try:
            cache_key = self._cache_key_extractor(data)
        except Exception as e:
            logger.error(
                f"{self.node_name}: Error extracting cache key for data type {type(data)}: {e}. "
                "Proceeding as if a cache miss and setting cache_key to None in context."
            )
            context["cache_hit"] = False
            context["cache_key"] = None  # Indicate key extraction failure
            return data

        context["cache_key"] = cache_key

        if cache_key in self._cache_store:
            cached_value = self._cache_store[cache_key]
            context["cache_hit"] = True
            logger.debug(f"{self.node_name}: Cache hit for key '{cache_key}'. Returning cached value.")
            return cached_value
        else:
            context["cache_hit"] = False
            logger.debug(f"{self.node_name}: Cache miss for key '{cache_key}'. Passing data through.")
            return data

    def store(self, key: Any, value: Any) -> None:
        """
        Explicitly stores a value in the cache under the given key.

        This method is designed to be called externally (e.g., by the orchestration
        layer or another specialized "writer" node) after a computation has been
        successfully performed following a cache miss.

        Args:
            key (Any): The unique key under which the `value` should be stored.
                       This key should ideally be the same one generated by the
                       `_cache_key_extractor` for the original input `data`.
            value (Any): The computed result or data to be cached.
        """
        if key is None:
            logger.warning(f"{self.node_name}: Attempted to store value with a None key. Store operation skipped.")
            return

        try:
            self._cache_store[key] = value
            logger.debug(f"{self.node_name}: Stored value for key '{key}'.")
        except Exception as e:
            logger.error(f"{self.node_name}: Failed to store value for key '{key}': {e}")