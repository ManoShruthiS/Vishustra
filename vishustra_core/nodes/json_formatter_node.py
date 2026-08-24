import json
import logging
from typing import Any, Dict, Optional

# Assuming BaseNode is available from this path in the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JsonFormatterNode(BaseNode):
    """
    A Vishustra node designed to format input data into a standardized JSON string.

    This node accepts various input types, including Python dictionaries, lists,
    and existing JSON strings. Its primary function is to ensure that the output
    is a valid JSON string, optionally pretty-printed with a specified indentation.

    If the input is an already-formatted JSON string, it will be parsed and then
    re-serialized to apply consistent formatting (e.g., adding indentation).
    For non-JSON-serializable Python objects or invalid JSON strings, appropriate
    errors are logged and propagated to facilitate robust error handling in the
    orchestration flow.
    """

    def __init__(self, indent: Optional[int] = None):
        """
        Initializes the JsonFormatterNode with an optional indentation level.

        Args:
            indent (Optional[int]): If provided, the JSON output will be
                                    pretty-printed with this many spaces of
                                    indentation. If set to `None`, the JSON
                                    will be output on a single line without
                                    extra whitespace.
        """
        self._indent = indent
        logger.debug(f"JsonFormatterNode initialized with indent={self._indent}")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "JsonFormatterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Formats the input data into a JSON string.

        The method handles different input types:
        - Dictionaries or lists: Directly serializes them into a JSON string.
        - Strings: Attempts to parse the string as JSON. If successful,
          it re-serializes the parsed object to apply the configured indentation.
          If parsing fails, a `ValueError` is raised.
        - Other types: A direct serialization attempt is made. If the type is
          not JSON-serializable (e.g., a custom object without a `__json__` method
          or a default serializer), a `TypeError` is raised.

        Args:
            data (Any): The input data to be formatted. Preferred types are
                        dictionaries, lists, or JSON strings.
            context (Dict[str, Any]): A dictionary containing contextual information
                                     relevant to the current processing step. This
                                     node does not directly use the context for
                                     formatting logic.

        Returns:
            str: A JSON-formatted string representation of the input data.

        Raises:
            ValueError: If the input 'data' is a string but does not contain
                        valid JSON.
            TypeError: If the input 'data' (of any type) cannot be serialized
                       into a valid JSON format.
        """
        formatted_json_string: str

        if isinstance(data, (dict, list)):
            try:
                formatted_json_string = json.dumps(data, indent=self._indent)
                logger.debug(f"{self.node_name}: Successfully serialized dict/list input.")
            except TypeError as e:
                logger.error(
                    f"{self.node_name}: Input data of type '{type(data).__name__}' is not JSON serializable: {e}"
                )
                raise TypeError(
                    f"Data is not JSON serializable: {e}"
                ) from e
        elif isinstance(data, str):
            try:
                # First, parse the string to validate it and get the Python object
                parsed_data = json.loads(data)
                # Then, re-serialize it to apply the desired indentation
                formatted_json_string = json.dumps(parsed_data, indent=self._indent)
                logger.debug(f"{self.node_name}: Successfully parsed and re-serialized JSON string input.")
            except json.JSONDecodeError as e:
                logger.error(f"{self.node_name}: Input string is not valid JSON: {e}")
                raise ValueError(
                    f"Input string is not valid JSON: {e}"
                ) from e
            except TypeError as e:
                # This catches rare cases where json.loads might return an object
                # that subsequently json.dumps cannot handle, or other issues.
                logger.error(
                    f"{self.node_name}: Unexpected TypeError during re-serialization of parsed JSON string: {e}"
                )
                raise TypeError(
                    f"Failed to re-serialize valid JSON data: {e}"
                ) from e
        else:
            logger.warning(
                f"{self.node_name}: Input data is of an unexpected type ('{type(data).__name__}'). Attempting direct serialization."
            )
            try:
                formatted_json_string = json.dumps(data, indent=self._indent)
                logger.debug(f"{self.node_name}: Successfully serialized data of type '{type(data).__name__}'.")
            except TypeError as e:
                logger.error(
                    f"{self.node_name}: Input data of type '{type(data).__name__}' is not JSON serializable: {e}"
                )
                raise TypeError(
                    f"Data of type '{type(data).__name__}' is not JSON serializable: {e}"
                ) from e

        return formatted_json_string
