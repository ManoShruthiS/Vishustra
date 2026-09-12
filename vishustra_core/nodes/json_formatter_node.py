import json
import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JsonFormatterNode(BaseNode):
    """
    A processing node designed to format arbitrary data into a JSON string.

    This node provides functionality to convert Python objects (like dictionaries,
    lists, numbers, booleans, and strings) into their JSON string representation.
    If the input data is already a string, the node will first attempt to parse
    it as JSON. This allows for re-formatting of existing JSON strings (e.g.,
    to add pretty-printing). If the string cannot be parsed as valid JSON,
    it will be treated as a literal string to be encapsulated in JSON.

    Configuration Options (via `context` dictionary):
        - 'indent' (int, optional): Specifies the indentation level for JSON
                                    output. If provided as a non-negative integer,
                                    the JSON will be pretty-printed. Defaults
                                    to `None`, resulting in compact JSON output.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "JsonFormatter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Formats the input data into a JSON string, applying specified indentation.

        Args:
            data: The input data to be formatted. This can be any JSON-serializable
                  Python object (dict, list, str, int, float, bool, None) or a
                  string that might contain JSON.
            context: A dictionary containing contextual information and configuration
                     parameters. The 'indent' key can be used to control pretty-printing.

        Returns:
            A JSON formatted string representation of the input data.

        Raises:
            TypeError: If the input data (or its parsed form) is not JSON-serializable.
            Exception: For any other unexpected errors encountered during processing.
        """
        # Determine indentation level from context, defaulting to compact output
        indent_level = context.get('indent', None)
        if not isinstance(indent_level, int) or indent_level < 0:
            indent_level = None  # Ensure indent_level is None or a valid non-negative integer

        logger.debug(
            f"[{self.node_name}] Processing data of type '{type(data).__name__}' "
            f"with configured indent level: {indent_level}"
        )

        # Attempt to parse string data first to handle re-formatting existing JSON strings
        data_to_serialize = data
        if isinstance(data, str):
            try:
                data_to_serialize = json.loads(data)
                logger.debug(f"[{self.node_name}] Successfully parsed input string as JSON.")
            except json.JSONDecodeError:
                logger.warning(
                    f"[{self.node_name}] Input string could not be parsed as valid JSON. "
                    "Treating it as a literal string for serialization."
                )
                # If it's not valid JSON, we'll serialize the original string itself
                # This will result in a JSON string like '"my plain string"'
            except Exception as e:
                logger.error(
                    f"[{self.node_name}] An unexpected error occurred while attempting "
                    f"to parse input string as JSON: {e}"
                )
                # Keep original data in case of unexpected parsing errors

        try:
            # Serialize the processed data into a JSON string
            formatted_json = json.dumps(data_to_serialize, indent=indent_level)
            logger.info(f"[{self.node_name}] Data successfully formatted as JSON.")
            return formatted_json
        except TypeError as e:
            logger.error(
                f"[{self.node_name}] Failed to serialize data to JSON. "
                f"Data type: '{type(data_to_serialize).__name__}'. Error: {e}"
            )
            raise TypeError(
                f"Data of type '{type(data_to_serialize).__name__}' is not JSON serializable: {e}"
            ) from e
        except Exception as e:
            logger.critical(
                f"[{self.node_name}] An unhandled exception occurred during JSON formatting: {e}",
                exc_info=True
            )
            raise Exception(
                f"An unexpected error occurred in JsonFormatterNode: {e}"
            ) from e