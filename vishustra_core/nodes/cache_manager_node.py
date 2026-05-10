import logging
import hashlib
from typing import Any, Dict, Optional
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A specialized node within the Vishustra framework designed to handle 
    memoization of LLM outputs and intermediate pipeline states.
    
    This node facilitates 'get', 'set', and 'delete' operations against a 
    provided storage backend (e.g., Redis, in-memory dict, or disk cache) 
    passed via the orchestration context.
    """

    @property
    def node_name(self) -> str:
        """Returns the unique identifier for the Cache Manager node."""
        return "CacheManagerNode"

    def _generate_deterministic_key(self, data: Any) -> str:
        """
        Generates a SHA-256 hash to act as a unique identifier for the 
        input data if a custom key is not provided.
        """
        try:
            return hashlib.sha256(str(data).encode('utf-8')).hexdigest()
        except Exception as e:
            logger.error(f"[{self.node_name}] Key generation failed: {str(e)}")
            return "default_cache_key"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the data by interacting with the cache store based on 
        the specified action in the context.

        Args:
            data: The payload to be cached or the fallback value if retrieval fails.
            context: Dictionary containing control flags and the storage backend.
                - 'cache_action': str ('get', 'set', 'evict')
                - 'cache_key': Optional[str] (explicit key)
                - 'cache_store': Dict-like object for storage
                - 'cache_bypass': bool (force skip cache)

        Returns:
            The cached value on 'get' (if hit), or the input data on 'set'/'evict'.
        """
        if context.get("cache_bypass", False):
            logger.debug(f"[{self.node_name}] Cache bypass enabled.")
            return data

        # Extract storage and action from context
        store = context.get("cache_store")
        action = context.get("cache_action", "get").lower()
        
        if store is None:
            logger.warning(f"[{self.node_name}] No 'cache_store' found in context. Skipping caching logic.")
            return data

        # Determine the lookup key
        provided_key = context.get("cache_key")
        lookup_key = provided_key if provided_key else self._generate_deterministic_key(data)

        try:
            if action == "get":
                cached_result = store.get(lookup_key)
                if cached_result is not None:
                    logger.info(f"[{self.node_name}] Cache hit for key: {lookup_key[:12]}...")
                    return cached_result
                
                logger.info(f"[{self.node_name}] Cache miss for key: {lookup_key[:12]}...")
                return data

            elif action == "set":
                # Only store if data is not None to avoid caching empty results
                if data is not None:
                    store[lookup_key] = data
                    logger.info(f"[{self.node_name}] Successfully stored result under key: {lookup_key[:12]}...")
                return data

            elif action == "evict":
                if lookup_key in store:
                    del store[lookup_key]
                    logger.info(f"[{self.node_name}] Evicted key: {lookup_key[:12]} from store.")
                return data

            else:
                logger.error(f"[{self.node_name}] Unsupported cache_action: '{action}'. Returning raw data.")
                return data

        except Exception as e:
            logger.exception(f"[{self.node_name}] Critical failure during cache {action} operation: {str(e)}")
            # Fallback to returning input data to prevent pipeline breakage
            return data