import json
import logging
from typing import Any, Dict, Optional

# Assuming BaseNode lives here in the actual project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class JSONFormatterNode(BaseNode):
    """
    A processing node that takes any JSON-serializable data and formats it
    into a pretty-printed JSON string.

    This node is useful for standardizing the output format of data,
    making it human-readable, or preparing it for storage/transmission
    in a consistent JSON structure.
    """

    def __init__(self, indent_level: Optional[int] = 2):
        """
        Initializes the JSONFormatterNode.

        Args:
            indent_level (Optional[int]): The number of spaces to use for indentation
                                          when formatting the JSON string. Defaults to 2.
                                          Use `None` for the most compact representation
                                          (no indentation, no newlines).
        Raises:
            TypeError: If `indent_level` is not an integer or None.
            ValueError: If `indent_level` is a negative integer.
        """
        if indent_level is not None and not isinstance(indent_level, int):
            raise TypeError(f"indent_level must be an integer or None, got {type(indent_level).__name__}")
        if indent_level is not None and indent_level < 0:
            raise ValueError(f"indent_level must be a non-negative integer or None, got {indent_level}")
            
        self._indent_level: Optional[int] = indent_level
        logger.debug(f"JSONFormatterNode initialized with indent_level: {self._indent_level}")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "JSONFormatter"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data by attempting to format it as a pretty-printed
        JSON string.

        If the input `data` is already a string, it first attempts to parse it
        as JSON. If successful, it then re-formats it. If the string is not
        valid JSON, or if the data is not JSON-serializable (e.g., a Python
        object that cannot be converted to JSON), an error is logged and
        a `ValueError` is raised.

        Args:
            data (Any): The input data to be formatted. Can be a dictionary, list,
                        primitive type, or a JSON string.
            context (Dict[str, Any]): The current processing context, useful for
                                       logging node-specific identifiers.

        Returns:
            str: The formatted JSON string.

        Raises:
            ValueError: If the input data is `None`, not JSON-serializable, or
                        cannot be parsed as valid JSON.
        """
        # Retrieve a unique identifier for the current node instance from context for better logs
        node_id = context.get("node_id", self.node_name) 

        if data is None:
            logger.warning(f"[{node_id}] Received None as input data. Cannot format as JSON.")
            raise ValueError("Input data cannot be None for JSONFormatterNode processing.")

        data_to_serialize: Any = data

        if isinstance(data, str):
            try:
                # If the input is a string, attempt to parse it first to ensure it's valid JSON.
                # This handles cases where raw JSON strings need re-formatting.
                data_to_serialize = json.loads(data)
                logger.debug(f"[{node_id}] Successfully parsed input string as JSON for re-formatting.")
            except json.JSONDecodeError as e:
                logger.error(
                    f"[{node_id}] Input string is not valid JSON. "
                    f"Error: {e}. Original string (first 100 chars): '{data[:100]}...'"
                )
                raise ValueError("Input string is not valid JSON and cannot be formatted.") from e
            except Exception as e:
                # Catch any unexpected errors during JSON string parsing
                logger.error(
                    f"[{node_id}] An unexpected error occurred while parsing input string as JSON. "
                    f"Error: {type(e).__name__}: {e}. Original string (first 100 chars): '{data[:100]}...'"
                )
                raise ValueError("Failed to parse input string as JSON due to an unexpected error.") from e

        try:
            formatted_json = json.dumps(data_to_serialize, indent=self._indent_level)
            logger.debug(f"[{node_id}] Successfully formatted data as JSON.")
            return formatted_json
        except TypeError as e:
            logger.error(
                f"[{node_id}] Input data of type '{type(data_to_serialize).__name__}' is not JSON-serializable. "
                f"Error: {e}. Data: {str(data_to_serialize)[:100]}..."
            )
            raise ValueError("Input data is not JSON-serializable and cannot be formatted.") from e
        except Exception as e:
            # Catch any other unexpected errors during JSON serialization
            logger.error(
                f"[{node_id}] An unexpected error occurred during JSON serialization. "
                f"Error: {type(e).__name__}: {e}. Data type: {type(data_to_serialize).__name__}"
            )
            raise ValueError("Failed to serialize data to JSON due to an unexpected error.") from e