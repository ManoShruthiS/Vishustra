import json
import logging
from typing import Any, Dict

# Assuming BaseNode is available at this path for the project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JSONFormatterNode(BaseNode):
    """
    A processing node that formats input data into a standardized JSON string.

    This node handles various input types:
    - If the input is a valid JSON string, it will be parsed and then re-serialized
      with standard indentation for readability (e.g., `indent=2`).
    - If the input is a Python dictionary, list, number, boolean, or None,
      it will be directly serialized to a JSON string with indentation.
    - If the input is a string that is NOT valid JSON, it will be treated as
      a literal string value and encased in a JSON string (e.g., "hello" becomes '"hello"').
    - If the input is an object that cannot be serialized to JSON, an error will be logged
      and a ValueError will be raised.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "json_formatter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Formats the input data into a pretty-printed JSON string.

        Args:
            data: The input data to be formatted. Can be any Python type
                  that is typically JSON-serializable or a JSON string.
            context: A dictionary containing contextual information for the node.
                     Not directly used by this node but passed along to adhere to
                     the BaseNode interface.

        Returns:
            A string representing the formatted JSON.

        Raises:
            ValueError: If the input data cannot be serialized into a valid JSON string.
        """
        formatted_json_string: str

        try:
            if isinstance(data, str):
                try:
                    # Attempt to parse if it's already a JSON string
                    parsed_data = json.loads(data)
                    formatted_json_string = json.dumps(parsed_data, indent=2)
                    logger.debug("Successfully re-formatted valid JSON string input.")
                except json.JSONDecodeError:
                    # If it's a string but not valid JSON, treat it as a literal string value
                    # and serialize it as such. E.g., "hello" -> '"hello"'
                    formatted_json_string = json.dumps(data) # No indent for literal strings
                    logger.warning(
                        f"Input data is a string but not valid JSON. "
                        f"Serializing as a literal JSON string value. Data preview: {data[:50]}"
                    )
            else:
                # For non-string types (dict, list, int, bool, None, etc.),
                # directly serialize to JSON with pretty-printing.
                formatted_json_string = json.dumps(data, indent=2)
                logger.debug("Successfully formatted non-string data to JSON.")

        except TypeError as e:
            # Catch errors for types that json.dumps cannot handle (e.g., custom objects without serialization)
            logger.error(
                f"JSONFormatterNode failed to serialize data due to an unsupported type: {type(data)}. "
                f"Error: {e}. Data preview: {str(data)[:100]}"
            )
            raise ValueError(f"Data type {type(data)} is not JSON serializable.") from e
        except Exception as e:
            # Catch any other unexpected errors during the serialization process
            logger.error(
                f"An unexpected error occurred in JSONFormatterNode during serialization: {e}. "
                f"Data preview: {str(data)[:100]}"
            )
            raise ValueError(f"Failed to format data to JSON: {e}") from e

        return formatted_json_string