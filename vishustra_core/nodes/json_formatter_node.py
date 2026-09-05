import json
import logging
from typing import Any, Dict

# Assuming BaseNode is available at this path as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class JSONFormatterNode(BaseNode):
    """
    A Vishustra node that formats input data into a JSON string.

    This node is designed to take any JSON-serializable Python object or a
    string containing valid JSON, and serialize it into a human-readable,
    indented JSON string. It's useful for standardizing data format or
    improving readability in logs or outputs.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "JSONFormatterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, formatting it as a JSON string.

        The `context` dictionary can specify 'indent' for json.dumps.
        Defaults to 2 if not specified.

        Args:
            data: The input data, which can be any JSON-serializable Python object
                  (e.g., dict, list, str, int, float, bool, None)
                  or a string containing valid JSON.
            context: A dictionary potentially containing formatting options.
                     Supported keys:
                       - 'indent' (int): The number of spaces to use for indentation
                                         in the output JSON string. Defaults to 2.
                                         Set to `None` for the most compact JSON.

        Returns:
            A string representing the JSON-formatted data.

        Raises:
            ValueError: If the input data is not JSON-serializable or
                        is a malformed JSON string that cannot be parsed.
        """
        indent_level = context.get('indent', 2)
        parsed_data = None

        if isinstance(data, str):
            try:
                # Attempt to parse the string as JSON first to ensure validity
                # and to allow reformatting of existing JSON strings.
                parsed_data = json.loads(data)
                logger.debug("Input data was a JSON string, successfully parsed for reformatting.")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode input string as JSON: '{data[:100]}...' Error: {e}")
                raise ValueError(
                    f"Input string is not valid JSON and cannot be formatted. Error: {e}"
                ) from e
        else:
            # Input is already a Python object (dict, list, int, etc.)
            parsed_data = data
            logger.debug(f"Input data is a Python object of type {type(data).__name__}, attempting direct serialization.")

        try:
            # Now, serialize the parsed (or original) Python object to a formatted JSON string
            formatted_json = json.dumps(parsed_data, indent=indent_level)
            logger.info("Data successfully formatted as JSON.")
            return formatted_json
        except TypeError as e:
            logger.error(f"Input data of type '{type(parsed_data).__name__}' is not JSON-serializable: {e}")
            raise ValueError(
                f"Input data of type '{type(parsed_data).__name__}' is not JSON-serializable. Error: {e}"
            ) from e
        except Exception as e:
            logger.exception(f"An unexpected error occurred during JSON serialization: {e}")
            raise RuntimeError(
                f"An unexpected error occurred while formatting data to JSON. Error: {e}"
            ) from e