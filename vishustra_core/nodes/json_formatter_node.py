
import json
import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class JsonFormatterNode(BaseNode):
    """
    A Vishustra processing node that formats input data into a JSON string.

    This node attempts to convert the input data into a Python dictionary or
    list if it's a JSON string, then serializes it into a human-readable
    JSON string. Non-serializable objects or invalid JSON strings will
    result in an error.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "JsonFormatterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by converting it into a formatted JSON string.

        Configuration options can be provided via the 'context' dictionary:
        - 'indent' (int | None): The indentation level for the JSON output.
                                 If None, a compact JSON string is produced.
                                 Defaults to 2 for readability.
        - 'sort_keys' (bool): If True, output dictionary keys will be sorted
                              alphabetically. Defaults to False.

        Args:
            data: The input data, which can be a dict, list, a JSON string,
                  or any other JSON-serializable Python object (e.g., int, bool, float).
            context: A dictionary containing execution context and configuration
                     parameters for this node.

        Returns:
            A formatted JSON string if the operation is successful.
            Returns None if the input data is not JSON-serializable or if
            an invalid JSON string was provided as input.
        """
        # Determine formatting options from context, with sensible defaults
        indent = context.get('indent', 2)
        sort_keys = context.get('sort_keys', False)

        # Prepare data for serialization
        serializable_data = data

        # If the input data is a string, try to parse it as JSON first
        if isinstance(data, str):
            try:
                serializable_data = json.loads(data)
            except json.JSONDecodeError as e:
                # If the string is not valid JSON, we cannot "format" it as JSON.
                # Log an error and indicate failure by returning None.
                logger.error(
                    f"[{self.node_name}] Failed to parse input string as valid JSON. "
                    f"Input sample: '{data[:200]}...', Error: {e}"
                )
                return None
            except Exception as e:
                # Catch any other unexpected errors during string parsing
                logger.error(
                    f"[{self.node_name}] An unexpected error occurred while parsing input string. "
                    f"Error: {e}", exc_info=True
                )
                return None

        # Attempt to serialize the (potentially parsed) Python object into a JSON string
        try:
            formatted_json = json.dumps(serializable_data, indent=indent, sort_keys=sort_keys)
            return formatted_json
        except TypeError as e:
            # Handle cases where the data contains non-JSON-serializable objects
            logger.error(
                f"[{self.node_name}] Encountered a TypeError during JSON serialization for data of type "
                f"'{type(serializable_data).__name__}'. Data might contain non-serializable objects. Error: {e}"
            )
            return None
        except Exception as e:
            # Catch any other unexpected exceptions during serialization
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during JSON serialization: {e}",
                exc_info=True
            )
            return None

