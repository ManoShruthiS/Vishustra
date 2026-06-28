import json
import logging
from typing import Any, Dict, Optional

# BaseNode is imported from the project's core module as specified.
# The actual content of BaseNode is defined in the project context.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JsonFormatterNode(BaseNode):
    """
    A Vishustra processing node that formats input data into a JSON string.

    This node intelligently handles various input types:
    - If the input `data` is a Python object (like dict, list, int, str, bool),
      it attempts to serialize it directly into a JSON string.
    - If the input `data` is already a string, it first tries to parse it as
      valid JSON. If successful, it then re-formats the parsed object.
      If the string is not valid JSON, it treats the string itself as a literal
      value to be serialized (e.g., the string "hello" becomes the JSON string '"hello"').

    Configuration options for JSON output like 'indent' (for pretty-printing)
    and 'sort_keys' can be provided during initialization.
    """

    def __init__(self, indent: Optional[int] = 2, sort_keys: bool = False):
        """
        Initializes the JsonFormatterNode with JSON formatting options.

        Args:
            indent: Controls JSON pretty-printing.
                    If a non-negative integer, JSON array elements and object
                    members will be pretty-printed with that indent level.
                    A value of `None` will produce a compact JSON string.
                    Defaults to 2 for readability.
            sort_keys: If `True`, the output of dictionaries will be sorted by key.
                       Defaults to `False`.
        """
        self._indent = indent
        self._sort_keys = sort_keys
        logger.debug(
            f"JsonFormatterNode initialized with indent='{self._indent}', "
            f"sort_keys='{self._sort_keys}'."
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "JsonFormatter"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data, attempting to format it as a JSON string.

        The `context` dictionary is passed through the pipeline but is not
        directly utilized by this specific node.

        Args:
            data: The input data to be formatted. This can be any Python object
                  that is JSON-serializable, or a string that may or may not
                  be valid JSON.
            context: A dictionary containing contextual information relevant
                     to the pipeline execution.

        Returns:
            A string containing the JSON-formatted representation of the data.

        Raises:
            ValueError: If the input data is fundamentally not JSON-serializable
                        (e.g., an object with circular references, or unsupported types).
            RuntimeError: For unexpected errors during the formatting process.
        """
        serializable_data = data

        if isinstance(data, str):
            try:
                # Attempt to parse string data as JSON for re-formatting.
                # If successful, the parsed object is used for final serialization.
                serializable_data = json.loads(data)
                logger.debug("Input data was a valid JSON string, parsed for re-formatting.")
            except json.JSONDecodeError:
                # If the string is not valid JSON, we treat the original string
                # itself as the value to be serialized (e.g., "invalid" -> '"invalid"').
                # This ensures the output is always valid JSON.
                logger.warning(
                    f"Input string is not valid JSON. Treating as literal string value. "
                    f"Truncated: '{data[:100]}...'" if len(data) > 100 else f"'{data}'"
                )
            except Exception as e:
                # Catch any other unexpected errors during json.loads
                logger.error(
                    f"Unexpected error encountered while attempting to parse string data: {e}",
                    exc_info=True
                )
                raise RuntimeError(f"Failed to process string data due to unexpected parsing error: {e}") from e

        try:
            # Serialize the (potentially parsed) data into a JSON string with configured options.
            formatted_json = json.dumps(
                serializable_data,
                indent=self._indent,
                sort_keys=self._sort_keys
            )
            logger.debug("Data successfully formatted to JSON.")
            return formatted_json
        except TypeError as e:
            # This handles cases where `serializable_data` contains non-JSON-serializable types.
            logger.error(f"Input data contains non-JSON-serializable types: {e}", exc_info=True)
            raise ValueError(f"Input data is not JSON-serializable: {e}") from e
        except Exception as e:
            # Catch any other unforeseen errors during json.dumps.
            logger.error(f"An unexpected error occurred during JSON serialization: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error during JSON formatting: {e}") from e
