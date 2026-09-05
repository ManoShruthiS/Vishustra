
import json
import logging
from typing import Any, Dict

# Assuming BaseNode is available at this path as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JsonFormatterNode(BaseNode):
    """
    A Vishustra processing node that serializes arbitrary input data into a consistent
    JSON string representation.

    This node ensures that the output is always a valid JSON string.
    If the input data is already a JSON string, it will be parsed and then
    re-serialized to ensure consistent formatting (e.g., indentation or
    canonical compact form). If the input data is a Python object (e.g., dict,
    list, primitive type), it will be directly serialized.

    Input data that is neither a valid JSON string nor a JSON-serializable
    Python object will result in an error.

    Context Parameters:
    - 'indent' (Optional[int]): If provided, the JSON output will be pretty-printed
                                 with the specified indentation level. If omitted,
                                 the output will be a compact JSON string.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "JsonFormatter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by formatting it into a JSON string.

        Args:
            data (Any): The input data to be formatted. This can be a JSON string,
                        a Python dictionary, list, or any JSON-serializable type.
            context (Dict[str, Any]): A dictionary of contextual information, which
                                       can include 'indent' for pretty-printing.

        Returns:
            Any: A string containing the formatted JSON.

        Raises:
            ValueError: If the input data is a string but not valid JSON.
            TypeError: If the input data is a Python object that cannot be
                       JSON serialized.
            RuntimeError: For any other unexpected errors during processing.
        """
        indent_level = context.get('indent', None)

        try:
            # Attempt to convert the input `data` into a Python object if it's a string.
            # This step is crucial for re-formatting existing JSON strings.
            python_object = data
            if isinstance(data, str):
                try:
                    python_object = json.loads(data)
                    logger.debug("JsonFormatterNode: Successfully parsed input string into Python object for re-formatting.")
                except json.JSONDecodeError as e:
                    logger.error(f"JsonFormatterNode: Input string is not valid JSON. Error: {e}", exc_info=True)
                    raise ValueError("Input data is a string but not valid JSON.") from e
                except TypeError as e:
                    # This should be highly unlikely for json.loads(str), but included for robustness.
                    logger.error(f"JsonFormatterNode: Unexpected TypeError during JSON parsing of string. Error: {e}", exc_info=True)
                    raise TypeError("Failed to parse JSON string due to unexpected type issue.") from e

            # Serialize the Python object (either original or parsed) into a JSON string.
            formatted_json_string = json.dumps(python_object, indent=indent_level)
            logger.debug("JsonFormatterNode: Successfully formatted data into JSON string.")
            return formatted_json_string

        except TypeError as e:
            logger.error(f"JsonFormatterNode: Data of type {type(data)} is not JSON serializable. Error: {e}", exc_info=True)
            raise TypeError(f"Input data of type '{type(data).__name__}' is not JSON serializable.") from e
        except Exception as e:
            # Catch any other unexpected errors during the formatting process
            logger.critical(f"JsonFormatterNode: An unexpected error occurred during JSON formatting. Error: {e}", exc_info=True)
            raise RuntimeError("An unexpected error occurred during JSON formatting.") from e

