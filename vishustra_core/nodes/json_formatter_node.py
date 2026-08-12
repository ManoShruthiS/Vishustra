import json
import logging
from typing import Any, Dict, Optional

# Assuming this path exists in the project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JsonFormatterNode(BaseNode):
    """
    A Vishustra processing node that formats input data into a consistent JSON string.

    This node takes any input data, attempts to convert it into a Python object if
    it's a JSON string, and then serializes the resulting object into a
    well-formatted JSON string. This ensures consistent output, optionally
    with pretty-printing.
    """

    def __init__(self, indent: Optional[int] = 2):
        """
        Initializes the JsonFormatterNode.

        Args:
            indent (Optional[int]): The indent level for pretty-printing JSON.
                                    If None, the JSON will be compact. Defaults to 2.
                                    This can be overridden per-call via the context.
        """
        self._indent = indent
        logger.debug(f"JsonFormatterNode initialized with default indent: {indent}")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "JSON Formatter"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data, attempting to convert it into a formatted JSON string.

        If the input `data` is a string, the node first attempts to parse it as JSON.
        If successful, the parsed Python object is then re-serialized. If parsing fails,
        the original string itself is treated as the data to be serialized (e.g.,
        the string "hello" would become the JSON string "\"hello\"").

        Args:
            data (Any): The input data to be formatted as JSON. This can be any
                        JSON-serializable Python object (dict, list, str, int, float, bool, None)
                        or a string that may or may not be valid JSON.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing. This node supports an optional
                                       `json_formatter_indent` key to override the
                                       configured indent level for the current process call.

        Returns:
            str: A JSON formatted string representation of the input data.

        Raises:
            ValueError: If the input data is not JSON serializable.
            RuntimeError: For unexpected errors during JSON serialization.
        """
        data_to_serialize = data

        if isinstance(data, str):
            try:
                # Attempt to parse string input to ensure validity and consistent re-serialization
                # (e.g., to apply pretty-printing even if input was compact JSON).
                data_to_serialize = json.loads(data)
                logger.debug("Input data was a string, successfully parsed as JSON for re-serialization.")
            except json.JSONDecodeError:
                # If the string cannot be parsed as JSON, treat the string itself as the
                # value to be serialized (e.g., "hello world" will become "\"hello world\"" in JSON).
                logger.warning(
                    "Input string could not be decoded as JSON. Serializing the raw string value."
                )
                # data_to_serialize remains the original string 'data' in this case
            except Exception as e:
                logger.error(
                    f"Unexpected error occurred when attempting to parse string input as JSON: {e}",
                    exc_info=True
                )
                # Fallback: proceed with the original string, letting json.dumps handle it.

        try:
            # Determine the indent level: prefer context override, then instance default
            current_indent = context.get("json_formatter_indent", self._indent)
            json_string = json.dumps(data_to_serialize, indent=current_indent)
            logger.info("Data successfully formatted to JSON string.")
            return json_string
        except TypeError as e:
            logger.error(
                f"Failed to serialize data to JSON due to a TypeError: {e}. Data might not be serializable.",
                exc_info=True
            )
            raise ValueError(f"Input data is not JSON serializable: {e}")
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during JSON serialization: {e}",
                exc_info=True
            )
            raise RuntimeError(f"JSON serialization failed unexpectedly: {e}")