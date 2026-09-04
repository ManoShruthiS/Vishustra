
import json
import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JsonFormatterNode(BaseNode):
    """
    A Vishustra node that formats input data into a JSON string with customizable options.

    This node takes any Python object that is serializable to JSON
    (e.g., dict, list, int, string, etc.) and converts it into a
    JSON string. If the input data is already a string, it attempts
    to parse it as JSON first. If successful, it then re-serializes
    the parsed object with the specified formatting options. If the
    string cannot be parsed as valid JSON, it logs a warning and
    returns the original string, as its structure cannot be formatted.

    JSON formatting options such as `indent`, `sort_keys`, `separators`, etc.,
    can be provided via the `context` dictionary under the key
    `'formatter_options'`. Only valid `json.dumps` keyword arguments
    will be utilized.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "JsonFormatterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Formats the input data into a JSON string using options from context.

        Args:
            data: The input data, which can be any JSON-serializable Python object.
                  If a string, it will attempt to parse as JSON first to allow
                  re-formatting of existing JSON structures.
            context: A dictionary containing operational context, including
                     optional 'formatter_options' for `json.dumps` arguments
                     (e.g., {'indent': 2, 'sort_keys': True}).

        Returns:
            A JSON formatted string, or the original data if formatting fails
            because the input was a non-JSON string.

        Raises:
            TypeError: If the input data (or its parsed form) is not JSON serializable.
            Exception: For other unexpected errors during JSON serialization.
        """
        # Extract and filter formatting options for json.dumps
        format_options = context.get('formatter_options', {})
        
        # Define valid kwargs for json.dumps to prevent passing unexpected arguments
        valid_json_dumps_kwargs = {
            'skipkeys', 'ensure_ascii', 'check_circular', 'allow_nan',
            'cls', 'indent', 'separators', 'default', 'sort_keys'
        }
        
        # Filter context options to include only valid json.dumps kwargs
        filtered_options = {k: v for k, v in format_options.items() if k in valid_json_dumps_kwargs}

        obj_to_serialize = data

        if isinstance(data, str):
            try:
                # If the input is a string, attempt to parse it as JSON first.
                # This allows re-formatting of an already JSON-encoded string.
                obj_to_serialize = json.loads(data)
                logger.debug(f"Input string successfully parsed as JSON for re-formatting.")
            except json.JSONDecodeError:
                # If the string is not valid JSON, we cannot apply structural formatting.
                # Log a warning and return the original string.
                logger.warning(
                    f"Input string is not valid JSON and cannot be structured-formatted. "
                    f"Returning original string. Data snippet: '{data[:200]}'"
                )
                return data
            except Exception as e:
                # Catch any other unexpected errors during string parsing
                logger.error(
                    f"Unexpected error while attempting to parse input string as JSON: {e}. "
                    f"Returning original string. Data snippet: '{data[:200]}'"
                )
                return data

        try:
            # Serialize the (potentially parsed) object into a JSON string
            json_string = json.dumps(obj_to_serialize, **filtered_options)
            logger.debug(f"Data successfully formatted as JSON.")
            return json_string
        except TypeError as e:
            # This error occurs if the Python object itself cannot be serialized to JSON
            logger.error(f"Input data (or its parsed form) is not JSON serializable: {e}. "
                         f"Data type: {type(obj_to_serialize)}. Value: {str(obj_to_serialize)[:200]}")
            raise # Re-raise to signal a critical issue with data content
        except Exception as e:
            # Catch any other unexpected serialization errors
            logger.error(f"An unexpected error occurred during JSON serialization: {e}. "
                         f"Data type: {type(obj_to_serialize)}. Value: {str(obj_to_serialize)[:200]}")
            raise # Re-raise to signal a critical issue
