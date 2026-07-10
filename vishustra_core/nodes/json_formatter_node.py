import json
import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JSONFormatterNode(BaseNode):
    """
    A Vishustra processing node that formats input data into a JSON string.
    
    This node intelligently handles various input types:
    - If the input `data` is already a Python dictionary or list, it directly
      serializes it into a JSON string.
    - If the input `data` is a string, it attempts to parse it as JSON. If
      successful, the parsed object is then serialized.
    - If the input `data` is a primitive type (int, float, bool) or another
      JSON-serializable Python object, it attempts direct serialization.
    
    Configuration for JSON serialization, such as indentation for pretty-printing,
    can be provided via the `context` dictionary.
    """
    
    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "JSONFormatterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data, converting it into a formatted JSON string.
        
        Args:
            data: The input data to be formatted. This can be a Python object
                  (e.g., dict, list, int, str, bool) that is JSON-serializable,
                  or a string representing a JSON object/array.
            context: A dictionary containing operational context for the node.
                     Supports 'json_indent' (int) to specify indentation levels
                     for pretty-printing. If not provided or None, compact JSON
                     is generated. Example: `{'json_indent': 4}`.
                     
        Returns:
            A string representing the JSON-formatted data.
            
        Raises:
            ValueError: If the input data is `None`, or if a string input
                        cannot be parsed as valid JSON, or if the data
                        cannot be serialized into JSON.
        """
        parsed_data: Any = None
        
        # Determine indentation level from context for pretty-printing
        indent_level = context.get('json_indent')
        if not isinstance(indent_level, (int, type(None))):
            logger.warning(
                f"[{self.node_name}] Invalid 'json_indent' type in context. "
                f"Expected int or None, but received {type(indent_level).__name__}. Ignoring and using compact format."
            )
            indent_level = None # Reset to default behavior (compact) if type is incorrect

        if data is None:
            logger.error(f"[{self.node_name}] Received None as input data. Cannot format to JSON.")
            raise ValueError(f"[{self.node_name}] Input data cannot be None.")
        elif isinstance(data, (dict, list)):
            # Data is already a dictionary or list, ready for serialization
            parsed_data = data
        elif isinstance(data, str):
            # Attempt to parse the string as JSON
            try:
                parsed_data = json.loads(data)
                logger.debug(f"[{self.node_name}] Successfully parsed input string as JSON.")
            except json.JSONDecodeError as e:
                logger.error(
                    f"[{self.node_name}] Failed to decode input string as JSON: {e}. "
                    f"Input sample: '{data[:200]}{'...' if len(data) > 200 else ''}'"
                )
                raise ValueError(
                    f"[{self.node_name}] Input string is not a valid JSON format."
                ) from e
        else:
            # For other primitive types (int, float, bool) or custom objects
            # that json.dumps can handle directly.
            logger.debug(
                f"[{self.node_name}] Input data is of type '{type(data).__name__}'. "
                f"Attempting direct JSON serialization."
            )
            parsed_data = data

        try:
            # Serialize the processed data into a JSON string
            json_string = json.dumps(parsed_data, indent=indent_level)
            logger.debug(
                f"[{self.node_name}] Successfully formatted data to JSON. "
                f"Indent level: {indent_level if indent_level is not None else 'compact'}."
            )
            return json_string
        except TypeError as e:
            logger.error(
                f"[{self.node_name}] Data of type '{type(parsed_data).__name__}' "
                f"is not JSON serializable: {e}. "
                f"Data sample: '{str(parsed_data)[:200]}{'...' if len(str(parsed_data)) > 200 else ''}'"
            )
            raise ValueError(
                f"[{self.node_name}] Input data is not JSON serializable."
            ) from e
        except Exception as e:
            logger.critical(
                f"[{self.node_name}] An unexpected and unhandled error occurred during JSON serialization: {e}",
                exc_info=True
            )
            raise RuntimeError(
                f"[{self.node_name}] An unexpected error prevented JSON formatting."
            ) from e