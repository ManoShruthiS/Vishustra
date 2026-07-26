import logging
import json
from typing import Any, Dict

# Assuming BaseNode is available at this path as per instructions
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JSONFormatterNode(BaseNode):
    """
    A Vishustra processing node that formats input data into a JSON string.

    This node attempts to convert various Python data types into a JSON formatted
    string. If the input is already a string, it first tries to parse it as JSON
    to ensure it's valid, then re-serializes it. Formatting options like 'indent'
    can be passed via the context.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "JSONFormatter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, converting it into a JSON formatted string.

        Args:
            data: The input data to be formatted. Can be a string (JSON or plain)
                  or a standard Python object (dict, list, int, float, bool, None).
            context: A dictionary for node-specific parameters.
                     Expected keys:
                       - 'indent' (int, optional): The number of spaces to use for
                                                   indentation in the JSON output.
                                                   Defaults to None (compact output).

        Returns:
            A JSON formatted string representation of the input data.

        Raises:
            ValueError: If the input data cannot be serialized into JSON.
        """
        indent = context.get('indent', None)
        data_to_serialize = data

        if isinstance(data, str):
            try:
                # Attempt to parse the string to ensure it's valid JSON.
                # If successful, we re-serialize the parsed object.
                parsed_object = json.loads(data)
                data_to_serialize = parsed_object
                logger.debug(f"JSONFormatterNode: Successfully parsed input string as JSON.")
            except json.JSONDecodeError as e:
                logger.warning(
                    f"JSONFormatterNode: Input string could not be parsed as valid JSON. "
                    f"Treating it as a literal string to be dumped. Error: {e}"
                )
                # If it's a non-JSON string, it will be dumped as a JSON string literal (e.g., "hello" -> "\"hello\"")

        try:
            formatted_json_string = json.dumps(data_to_serialize, indent=indent)
            logger.info(f"JSONFormatterNode: Successfully formatted data into JSON string.")
            return formatted_json_string
        except TypeError as e:
            logger.error(
                f"JSONFormatterNode: Failed to serialize data into JSON due to a non-serializable type. "
                f"Input data type: {type(data)} (after potential parsing to {type(data_to_serialize)}). Error: {e}"
            )
            raise ValueError(f"Data cannot be serialized to JSON: {e}") from e
