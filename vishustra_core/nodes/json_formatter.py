import json
import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JsonFormatterNode(BaseNode):
    """
    A processing node that robustly formats input data into a JSON string.

    This node handles various input types:
    - If the input `data` is already a string, it first attempts to parse it
      as a JSON document. If successful, the parsed Python object is then
      re-serialized. If parsing fails (i.e., the string is not valid JSON),
      the original string itself is treated as a literal value to be
      JSON-encoded (e.g., "hello" becomes "\"hello\"").
    - If the input `data` is not a string (e.g., dict, list, int, bool, None),
      it is directly serialized to a JSON string.

    Context parameters can influence the formatting:
        - 'indent' (int, optional): The number of spaces to use for indentation
                                    when pretty-printing the JSON output.
                                    If omitted or invalid, the JSON will be
                                    output in a compact representation.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "JsonFormatter"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Formats the input data into a JSON string, applying optional pretty-printing.

        Args:
            data (Any): The input data to be formatted. This can be a Python object
                        (like dict, list, int, etc.) or a string.
            context (Dict[str, Any]): A dictionary of operational parameters.
                                      Expected key: 'indent' (int).

        Returns:
            str: The JSON formatted string representation of the input data.

        Raises:
            TypeError: If the input data (or its parsed form) contains elements
                       that are not JSON serializable (e.g., a `datetime` object
                       without a custom encoder).
            Exception: For any other unexpected errors during the formatting process.
        """
        # Retrieve and sanitize the 'indent' parameter from the context
        indent = context.get('indent')
        if not isinstance(indent, int) or indent < 0:
            indent = None  # Default to compact JSON if indent is not a valid non-negative integer

        data_to_serialize: Any

        if isinstance(data, str):
            try:
                # Attempt to parse the string as JSON
                data_to_serialize = json.loads(data)
                logger.debug("Input string successfully parsed as JSON document.")
            except json.JSONDecodeError:
                # If the string is not valid JSON, treat it as a literal string value
                logger.warning(
                    f"Input string is not a valid JSON document. Treating it as a literal string "
                    f"to be JSON-encoded. Snippet: '{data[:100]}{'...' if len(data) > 100 else ''}'"
                )
                data_to_serialize = data
            except Exception as e:
                # Catch any other unexpected errors during string parsing
                logger.error(
                    f"An unexpected error occurred while attempting to parse input string as JSON: {e}",
                    exc_info=True
                )
                raise TypeError(f"Failed to process input string for JSON formatting: {e}") from e
        else:
            # For non-string inputs, use the data directly
            data_to_serialize = data

        try:
            # Serialize the prepared data into a JSON string
            formatted_json_string = json.dumps(data_to_serialize, indent=indent)
            logger.debug(f"Data successfully serialized to JSON string.")
            return formatted_json_string
        except TypeError as e:
            logger.error(
                f"Failed to serialize data of type '{type(data_to_serialize).__name__}' to JSON. "
                f"Reason: {e}", exc_info=True
            )
            raise TypeError(
                f"Input data contains non-JSON serializable elements "
                f"(e.g., datetime objects or custom classes without a custom encoder): {e}"
            ) from e
        except Exception as e:
            # Catch any other unexpected errors during the final serialization
            logger.error(
                f"An unexpected error occurred during JSON serialization of data. Reason: {e}",
                exc_info=True
            )
            raise Exception(f"Unhandled error during JSON formatting: {e}") from e