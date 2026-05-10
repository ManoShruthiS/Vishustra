import json
import logging
from typing import Any, Dict
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JsonFormatterNode(BaseNode):
    """
    A utility node designed to format raw input into a standardized JSON string.
    
    This node handles dictionary, list, and string inputs. If the input is a string,
    it attempts to parse it as JSON first to ensure structural integrity before 
    applying pretty-printing.
    """

    @property
    def node_name(self) -> str:
        """Returns the unique identifier for the JSON formatting node."""
        return "JsonFormatterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes input data into a prettified JSON string.

        Args:
            data (Any): The payload to be formatted. Can be a dict, list, or string.
            context (Dict[str, Any]): Configuration context. 
                Supported keys: 
                - 'indent' (int): Spaces for indentation. Defaults to 4.
                - 'sort_keys' (bool): Whether to sort dictionary keys. Defaults to True.

        Returns:
            str: A formatted JSON string.

        Raises:
            ValueError: If the data cannot be serialized to JSON.
        """
        indent = context.get("json_indent", 4)
        sort_keys = context.get("json_sort_keys", True)

        try:
            # If input is a string, verify if it's already a JSON string to avoid double-escaping
            if isinstance(data, str):
                try:
                    processed_data = json.loads(data)
                    logger.debug("Successfully parsed input string as JSON before re-formatting.")
                except json.JSONDecodeError:
                    # Input is a plain string, keep it as is for serialization
                    logger.debug("Input string is not a JSON object; treating as raw text.")
                    processed_data = data
            else:
                processed_data = data

            # Serialize to formatted string
            formatted_json = json.dumps(
                processed_data, 
                indent=indent, 
                sort_keys=sort_keys,
                ensure_ascii=False
            )
            
            return formatted_json

        except (TypeError, OverflowError) as e:
            logger.error(f"Serialization failed in {self.node_name}: {str(e)}")
            raise ValueError(f"Data provided to {self.node_name} is not JSON serializable: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in {self.node_name}: {str(e)}")
            raise RuntimeError(f"An internal error occurred during JSON formatting: {e}")

```python
# Usage Example (Internal Documentation):
# node = JsonFormatterNode()
# result = node.process({"key": "value"}, {"json_indent": 2})
