import json
import logging
from typing import Any, Dict, Optional
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JSONFormatterNode(BaseNode):
    """
    A node responsible for converting input data into a standardized, 
    formatted JSON string. Supports custom indentation and key sorting 
    via the processing context.
    """

    @property
    def node_name(self) -> str:
        """Returns the canonical name of the JSON formatting node."""
        return "JSONFormatterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Transforms the input data into a formatted JSON string.
        
        The method attempts to parse string input as JSON if it isn't already 
        a dictionary or list, then re-serializes it based on context parameters.

        Args:
            data (Any): The data to be formatted. Expected to be a dict, list, or JSON string.
            context (Dict[str, Any]): Configuration for formatting. 
                Supported keys: 
                - 'indent': int (default: 4)
                - 'sort_keys': bool (default: True)
                - 'ensure_ascii': bool (default: False)

        Returns:
            str: The formatted JSON string.

        Raises:
            ValueError: If the input data cannot be serialized to JSON.
        """
        indent = context.get("indent", 4)
        sort_keys = context.get("sort_keys", True)
        ensure_ascii = context.get("ensure_ascii", False)

        processed_data = data

        # If data is a string, attempt to load it first to ensure valid structure
        if isinstance(data, str):
            try:
                processed_data = json.loads(data)
                logger.debug("Input data identified as string; successfully parsed into JSON object.")
            except json.JSONDecodeError as e:
                logger.warning(f"Input data is a string but not valid JSON: {str(e)}. Proceeding with raw string serialization.")

        try:
            formatted_json = json.dumps(
                processed_data,
                indent=indent,
                sort_keys=sort_keys,
                ensure_ascii=ensure_ascii
            )
            logger.info("Successfully formatted data into JSON.")
            return formatted_json

        except (TypeError, OverflowError) as e:
            logger.error(f"Failed to serialize data to JSON in {self.node_name}: {str(e)}")
            raise ValueError(f"Data provided to {self.node_name} is not JSON serializable: {str(e)}") from e

    def __repr__(self) -> str:
        return f"<{self.node_name}()>"