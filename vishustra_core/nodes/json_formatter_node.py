import json
import logging
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JSONFormatterNode(BaseNode):
    """
    A Vishustra processing node that formats input data into a JSON string.

    This node can take either a JSON string (which it will parse and re-serialize
    for consistent formatting) or a Python object (like dict, list, int, str, etc.)
    and convert it into a formatted JSON string.

    Configuration for indentation and key sorting can be provided during initialization
    or overridden dynamically via the 'context' dictionary.
    """

    def __init__(self, indent: Optional[int] = 2, sort_keys: bool = False):
        """
        Initializes the JSONFormatterNode.

        Args:
            indent: The indentation level for pretty-printing JSON. Use None for
                    a compact JSON string. Defaults to 2.
            sort_keys: If True, the output of dictionaries will be sorted by key.
                       Defaults to False for potentially better performance and
                       preservation of insertion order in modern Python.
        """
        self._indent = indent
        self._sort_keys = sort_keys
        logger.debug(
            f"JSONFormatterNode initialized with default indent={self._indent}, sort_keys={self._sort_keys}"
        )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "JSONFormatter"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data, ensuring it is a valid JSON structure (either
        initially a string or a Python object) and formatting it according
        to the node's configuration.

        Args:
            data: The input data. This can be a JSON string or any Python object
                  that is naturally JSON-serializable (e.g., dict, list, int,
                  float, bool, str, None).
            context: A dictionary containing runtime information and potential
                     overrides for formatting options.
                     Recognized context keys:
                     - 'json_formatter_indent': Overrides the node's `indent` level
                                                (int or None).
                     - 'json_formatter_sort_keys': Overrides the node's `sort_keys`
                                                   setting (bool).

        Returns:
            A formatted JSON string representation of the input data.

        Raises:
            ValueError: If the input data is a string but not valid JSON,
                        or if the input data cannot be serialized to JSON.
        """
        resolved_indent = context.get('json_formatter_indent', self._indent)
        resolved_sort_keys = context.get('json_formatter_sort_keys', self._sort_keys)

        python_object: Any
        if isinstance(data, str):
            try:
                # Attempt to parse the string to ensure it's valid JSON
                python_object = json.loads(data)
                logger.debug("Input data was a JSON string, successfully parsed for re-formatting.")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse input string as JSON: {e}. Data snippet: {data[:200]}...")
                raise ValueError("Input string is not valid JSON.") from e
        else:
            # If not a string, assume it's a Python object to be serialized directly
            python_object = data
            logger.debug(f"Input data was a Python object of type {type(data).__name__}, preparing for serialization.")

        try:
            # Serialize the Python object back to a formatted JSON string
            formatted_json_string = json.dumps(
                python_object,
                indent=resolved_indent,
                sort_keys=resolved_sort_keys
            )
            logger.info(
                f"Data successfully formatted to JSON with indent={resolved_indent}, "
                f"sort_keys={resolved_sort_keys}."
            )
            return formatted_json_string
        except TypeError as e:
            logger.error(
                f"Failed to serialize Python object to JSON: {e}. "
                f"Object type: {type(python_object).__name__}. Value: {repr(python_object)[:200]}..."
            )
            raise ValueError(
                "Input data could not be serialized to JSON. "
                "Ensure it contains only JSON-serializable types (e.g., dicts, lists, "
                "strings, numbers, booleans, None), or custom objects with a __json__ method."
            ) from e
        except Exception as e:
            # Catch any other unexpected serialization errors
            logger.critical(f"An unexpected error occurred during JSON serialization: {e}")
            raise RuntimeError(f"Unexpected error during JSON serialization: {e}") from e