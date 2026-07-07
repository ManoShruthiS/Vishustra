import logging
from typing import Any, Dict, Optional

# Assuming vishustra_core is accessible in the project structure
# For standalone execution, you might need to mock or define BaseNode locally.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class CacheManagerNode(BaseNode):
    """
    A Vishustra processing node that acts as an in-memory cache manager.
    It supports operations like setting, getting, deleting, and clearing
    key-value pairs within its internal cache.
    """

    def __init__(self, initial_cache: Optional[Dict[str, Any]] = None):
        """
        Initializes the CacheManagerNode with an optional initial cache state.
        """
        self._cache: Dict[str, Any] = initial_cache if initial_cache is not None else {}
        logger.info("CacheManagerNode initialized with %d initial items.", len(self._cache))

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "CacheManagerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes caching operations based on the provided data and context.

        Expected `context` format:
        `{'operation': 'GET' | 'SET' | 'DELETE' | 'CLEAR'}`

        Expected `data` format varies by operation:
        - For 'GET' and 'DELETE': `{'key': 'some_key'}`
        - For 'SET': `{'key': 'some_key', 'value': 'some_value'}`
        - For 'CLEAR': `{}` (empty dictionary or any data will clear the cache)

        Returns a dictionary indicating the result of the operation.
        """
        operation = context.get('operation')

        if not operation or not isinstance(operation, str):
            logger.error("Missing or invalid 'operation' in context for CacheManagerNode.")
            return {"status": "ERROR", "message": "Operation type not specified or invalid."}

        operation = operation.upper()
        cache_key = data.get('key') if isinstance(data, dict) else None

        try:
            if operation == 'GET':
                if cache_key is None:
                    logger.warning("GET operation attempted without a 'key' in data.")
                    return {"status": "ERROR", "message": "Key is required for GET operation."}

                if cache_key in self._cache:
                    value = self._cache[cache_key]
                    logger.debug(f"Cache HIT for key: '{cache_key}'")
                    return {"status": "SUCCESS", "operation": "GET", "key": cache_key, "value": value}
                else:
                    logger.debug(f"Cache MISS for key: '{cache_key}'")
                    return {"status": "NOT_FOUND", "operation": "GET", "key": cache_key, "value": None}

            elif operation == 'SET':
                if cache_key is None:
                    logger.warning("SET operation attempted without a 'key' in data.")
                    return {"status": "ERROR", "message": "Key is required for SET operation."}
                if 'value' not in data:
                    logger.warning(f"SET operation attempted for key '{cache_key}' without a 'value'.")
                    return {"status": "ERROR", "message": "Value is required for SET operation."}

                self._cache[cache_key] = data['value']
                logger.info(f"Cache SET for key: '{cache_key}'")
                return {"status": "SUCCESS", "operation": "SET", "key": cache_key}

            elif operation == 'DELETE':
                if cache_key is None:
                    logger.warning("DELETE operation attempted without a 'key' in data.")
                    return {"status": "ERROR", "message": "Key is required for DELETE operation."}

                if cache_key in self._cache:
                    del self._cache[cache_key]
                    logger.info(f"Cache DELETE for key: '{cache_key}'")
                    return {"status": "SUCCESS", "operation": "DELETE", "key": cache_key}
                else:
                    logger.warning(f"DELETE operation failed, key '{cache_key}' not found.")
                    return {"status": "NOT_FOUND", "operation": "DELETE", "key": cache_key, "message": "Key not found."}

            elif operation == 'CLEAR':
                initial_size = len(self._cache)
                self._cache.clear()
                logger.info(f"Cache CLEAR operation performed. {initial_size} items removed.")
                return {"status": "SUCCESS", "operation": "CLEAR", "items_cleared": initial_size}

            else:
                logger.error(f"Unsupported cache operation: '{operation}'")
                return {"status": "ERROR", "message": f"Unsupported operation: {operation}"}

        except Exception as e:
            logger.exception(f"An unexpected error occurred during cache operation '{operation}' for key '{cache_key}'.")
            return {"status": "ERROR", "message": f"An unexpected error occurred: {str(e)}"}
