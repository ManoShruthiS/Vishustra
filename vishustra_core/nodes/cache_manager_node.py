import logging
import hashlib
import json
from typing import Any, Dict, Optional
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A node responsible for managing stateful caching within the LLM orchestration pipeline.
    It supports lookup and storage operations to reduce redundant computations and API calls.
    """

    def __init__(self, external_storage: Optional[Dict[str, Any]] = None):
        """
        Initializes the CacheManagerNode.
        :param external_storage: Optional dictionary-like object to act as the cache store.
        """
        self._internal_cache = external_storage if external_storage is not None else {}

    @property
    def node_name(self) -> str:
        """Returns the unique identifier for this node type."""
        return "CacheManagerNode"

    def _generate_cache_key(self, data: Any) -> str:
        """
        Generates a deterministic SHA-256 hash for the provided input data.
        """
        try:
            # Sort keys to ensure consistent hashing for dictionary inputs
            serialized_data = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(serialized_data.encode("utf-8")).hexdigest()
        except (TypeError, ValueError) as e:
            logger.warning("Data serialization failed for key generation, falling back to string representation: %s", e)
            return hashlib.sha256(str(data).encode("utf-8")).hexdigest()

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by checking the cache or storing new results.
        
        Expected context keys:
        - cache_action: 'lookup' (default) or 'store'
        - cache_namespace: Optional string to isolate cache entries.
        - cache_payload: The result to be stored if action is 'store'.
        """
        action = context.get("cache_action", "lookup").lower()
        namespace = context.get("cache_namespace", "global")
        
        if namespace not in self._internal_cache:
            self._internal_cache[namespace] = {}

        try:
            cache_key = self._generate_cache_key(data)

            if action == "lookup":
                if cache_key in self._internal_cache[namespace]:
                    logger.info("Cache HIT for key: %s in namespace: %s", cache_key, namespace)
                    return {
                        "cache_status": "hit",
                        "data": self._internal_cache[namespace][cache_key]
                    }
                
                logger.info("Cache MISS for key: %s", cache_key)
                return {
                    "cache_status": "miss",
                    "data": data
                }

            elif action == "store":
                payload = context.get("cache_payload")
                if payload is None:
                    logger.error("Storage requested but 'cache_payload' is missing from context.")
                    return data
                
                self._internal_cache[namespace][cache_key] = payload
                logger.info("Successfully persisted data to cache for key: %s", cache_key)
                return payload

            else:
                logger.warning("Unrecognized cache action: '%s'. Returning original data.", action)
                return data

        except Exception as e:
            logger.error("Critical failure during cache processing: %s", str(e), exc_info=True)
            # Fallback to returning original data to prevent breaking the pipeline
            return data
            
    def clear_cache(self, namespace: Optional[str] = None) -> None:
        """
        Utility method to clear the cache.
        """
        if namespace and namespace in self._internal_cache:
            del self._internal_cache[namespace]
            logger.info("Cache namespace '%s' cleared.", namespace)
        else:
            self._internal_cache.clear()
            logger.info("Entire cache registry cleared.")

# end of file: cache_manager_node.py