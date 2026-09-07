import json
import logging
from typing import Any, Dict

# Assuming BaseNode is correctly available at this path in the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JsonFormatterNode(BaseNode):
    """
    A Vishustra node that serializes input data into a JSON string.

    This node takes any Python object and attempts to convert it into
    a JSON formatted string. It supports specifying the 'indent' level
    for pretty-printing via the processing context.

    Context Parameters:
        - 'json_indent' (int, optional): The indentation level for pretty-printing JSON.
                                         Defaults to None (compact format) if not provided.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "JsonFormatter"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data by serializing it into a JSON string.

        Args:
            data (Any): The input data to be serialized. This can be any
                        Python object that is JSON-serializable (e.g., dict, list,
                        string, number, boolean, None).
            context (Dict[str, Any]): A dictionary containing additional information
                                      or configuration for processing.
                                      Expected key: 'json_indent' (int).

        Returns:
            str: The JSON formatted string representation of the input data.

        Raises:
            ValueError: If the input data is not JSON-serializable.
            RuntimeError: For unexpected errors during serialization.
        """
        indent_level = context.get('json_indent')

        logger.debug(
            f"[{self.node_name}] Attempting to format data as JSON. Indent level: {indent_level}"
        )

        try:
            # Use json.dumps to serialize the data.
            # If indent_level is None, it defaults to a compact JSON string.
            # If it's an int, it pretty-prints with that indentation.
            formatted_json = json.dumps(data, indent=indent_level)
            logger.info(f"[{self.node_name}] Successfully formatted data as JSON.")
            return formatted_json
        except TypeError as e:
            error_msg = (
                f"[{self.node_name}] Failed to serialize data to JSON. "
                f"Data type '{type(data).__name__}' is not JSON-serializable. Error: {e}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        except Exception as e:
            # Catch any other unexpected errors during serialization
            error_msg = (
                f"[{self.node_name}] An unexpected error occurred during JSON serialization. Error: {e}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e