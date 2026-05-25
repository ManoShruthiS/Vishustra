import json
import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JsonFormatterNode(BaseNode):
    """
    A processing node that takes any JSON-serializable data and
    returns a standardized, pretty-printed JSON string.

    This node is useful for ensuring consistent JSON output,
    pretty-printing JSON structures, or validating JSON string inputs.
    It standardizes the output by re-serializing with a fixed indentation.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "JsonFormatter"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data, attempting to serialize it into a
        pretty-printed JSON string with an indent of 2 spaces.

        If the input `data` is already a string, this method first attempts
        to parse it to ensure validity as JSON. If successful, it then
        re-serializes the parsed data for consistent formatting. If `data`
        is not a string, it directly attempts to serialize it.

        Robust error handling is included for cases where data is not
        JSON-serializable or where a string input is not valid JSON.

        Args:
            data: The input data to be formatted. This can be any
                  JSON-serializable Python object (e.g., dict, list, str,
                  int, float, bool, None). If `data` is a string, it must
                  contain valid JSON content.
            context: A dictionary containing contextual information for the node.
                     This node currently does not utilize the context but
                     adheres to the `BaseNode` interface.

        Returns:
            A pretty-printed JSON string representation of the input data.

        Raises:
            ValueError: If the input data cannot be serialized to JSON
                        (e.g., contains unsupported types) or if a string
                        input is not valid JSON.
            RuntimeError: For any unexpected errors during processing.
        """
        try:
            # If the input data is a string, we first attempt to decode it.
            # This step validates the string's JSON structure and converts
            # it into a Python object (dict, list, etc.).
            # We then re-serialize this object to ensure consistent formatting.
            if isinstance(data, str):
                parsed_data = json.loads(data)
                return json.dumps(parsed_data, indent=2)
            else:
                # For non-string data (e.g., dict, list, int), directly serialize it.
                return json.dumps(data, indent=2)
        except json.JSONDecodeError as e:
            logger.error(
                f"JsonFormatterNode '{self.node_name}' failed to decode "
                f"input string as JSON. Error: {e}. "
                f"Data snippet: '{str(data)[:200]}'"
            )
            raise ValueError(f"Input string is not valid JSON: {e}") from e
        except TypeError as e:
            logger.error(
                f"JsonFormatterNode '{self.node_name}' failed to serialize "
                f"input data to JSON due to unsupported type. Error: {e}. "
                f"Data type: {type(data).__name__}. "
                f"Data snippet: '{str(data)[:200]}'"
            )
            raise ValueError(f"Input data is not JSON serializable: {e}") from e
        except Exception as e:
            logger.error(
                f"An unexpected error occurred in JsonFormatterNode '{self.node_name}' "
                f"during JSON processing. Error: {e}. "
                f"Data type: {type(data).__name__}. "
                f"Data snippet: '{str(data)[:200]}'"
            )
            raise RuntimeError(f"Unexpected error during JSON formatting: {e}") from e