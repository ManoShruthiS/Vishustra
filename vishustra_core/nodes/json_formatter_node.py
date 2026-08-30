import json
import logging
from typing import Any, Dict, Optional

# Assuming vishustra_core.nodes.base_node exists in the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class JSONFormatterNode(BaseNode):
    """
    A Vishustra processing node that formats input data into a JSON string.

    This node can take either a Python object (like a dict or list) or a JSON string
    as input. It ensures the output is a consistently formatted JSON string.
    If the input is already a JSON string, it will parse it and then re-serialize
    it to apply formatting (e.g., indentation).
    """

    def __init__(self, indent: Optional[int] = None):
        """
        Initializes the JSONFormatterNode.

        Args:
            indent (Optional[int]): If provided, JSON output will be pretty-printed
                                    with the specified indentation level.
                                    If None, the JSON will be compact.
        """
        self._indent = indent
        logger.debug(f"JSONFormatterNode initialized with indent={indent}")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "JSONFormatter"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data, converting it into a formatted JSON string.

        If the input `data` is a string, the node first attempts to parse it as JSON.
        If successful, or if `data` was already a Python object, it then serializes
        the resulting object into a JSON string, applying the configured indentation.
        If `data` is a string but not valid JSON, it's treated as a literal string
        value during serialization.

        Args:
            data (Any): The input data, which can be a Python object (dict, list, etc.)
                        or a JSON string.
            context (Dict[str, Any]): The current processing context, available
                                      for nodes that require additional state or parameters,
                                      but not directly used by this specific node.

        Returns:
            str: A formatted JSON string representation of the input data.

        Raises:
            TypeError: If the input data (or the parsed object from an input string)
                       is not JSON serializable.
            Exception: For other unexpected errors during processing.
        """
        parsed_data = None
        
        # If data is a string, attempt to parse it first.
        # This allows re-formatting existing JSON strings.
        if isinstance(data, str):
            try:
                parsed_data = json.loads(data)
                logger.debug(
                    "Input data was a valid JSON string. Parsed it for consistent re-serialization."
                )
            except json.JSONDecodeError as e:
                # If it's a string but not valid JSON, treat it as a literal string value
                logger.warning(
                    f"Input data is a string but not valid JSON. "
                    f"Treating it as a raw string value for serialization. Error: {e}"
                )
                parsed_data = data
        else:
            # Data is already a Python object (dict, list, int, etc.)
            parsed_data = data

        formatted_json: str
        try:
            # Serialize the parsed_data (or the original data if not a string) to JSON
            formatted_json = json.dumps(parsed_data, indent=self._indent)
            logger.info(
                f"Successfully formatted data as JSON (indent={'compact' if self._indent is None else self._indent})."
            )
        except TypeError as e:
            logger.error(
                f"Failed to serialize data to JSON due to a TypeError: {e}. "
                f"Input data type: {type(parsed_data)}. Data: {parsed_data!r}"
            )
            # Re-raise the exception as this is a critical failure for the node's purpose
            raise
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during JSON serialization: {e}. "
                f"Input data type: {type(parsed_data)}. Data: {parsed_data!r}"
            )
            raise

        return formatted_json