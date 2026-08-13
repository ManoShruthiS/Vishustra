import json
import logging
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JsonFormatterNode(BaseNode):
    """
    A processing node that formats input data into a JSON string.

    This node prioritizes treating string inputs as potential JSON content,
    attempting to parse them first to allow re-formatting existing JSON strings.
    If a string input is not valid JSON, or if the input is not a string,
    the node directly serializes the raw data. Non-JSON serializable Python
    objects will result in a `ValueError`.
    """

    def __init__(self, indent: Optional[int] = 2):
        """
        Initializes the JsonFormatterNode.

        Args:
            indent: The indentation level for pretty-printing the JSON output.
                    Set to None for the most compact JSON representation.
                    Defaults to 2 spaces.
        """
        self._indent = indent
        logger.debug(f"JsonFormatterNode initialized with indent: {indent}")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "JsonFormatter"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Formats the input data into a JSON string, applying specified indentation.

        Args:
            data: The input data to be formatted. This can be any JSON-serializable
                  Python object, or a string that may or may not contain valid JSON.
            context: A dictionary containing contextual information for processing.
                     This node does not directly use the context but it is
                     part of the BaseNode API.

        Returns:
            A string representing the formatted JSON output.

        Raises:
            ValueError: If the input data (after potential parsing) is not
                        JSON serializable.
            RuntimeError: For unexpected failures during JSON serialization.
        """
        # Determine the actual Python object to serialize.
        # If 'data' is a string, we attempt to parse it first to allow re-formatting
        # existing JSON strings, otherwise, we serialize the string literal itself.
        object_to_serialize: Any = data

        if isinstance(data, str):
            try:
                object_to_serialize = json.loads(data)
                logger.debug("Input string successfully parsed as JSON for re-formatting.")
            except json.JSONDecodeError:
                logger.debug(
                    "Input string is not valid JSON. Proceeding to serialize the string itself."
                    " The output will be a JSON string representing the original string literal."
                )
                # If it's not valid JSON, 'object_to_serialize' remains the original string.
                # 'json.dumps' will then serialize this string as '"original string"'.
            except Exception as e:
                # Catch any other unexpected errors during string parsing
                logger.warning(
                    f"Unexpected error encountered while attempting to parse string data as JSON: {e}."
                    " Proceeding with the raw string literal."
                )
                object_to_serialize = data # Fallback to original string

        try:
            formatted_json_string = json.dumps(object_to_serialize, indent=self._indent)
            logger.info("Data successfully formatted to JSON string.")
            return formatted_json_string
        except TypeError as e:
            logger.error(
                f"Input data is not JSON serializable: {e}. "
                f"Attempted to serialize type: {type(object_to_serialize).__name__}"
            )
            raise ValueError(f"Data provided to JsonFormatter is not JSON serializable: {e}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during JSON serialization: {e}")
            raise RuntimeError(f"Failed to format data to JSON due to an internal error: {e}") from e