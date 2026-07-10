import json
import logging
from typing import Any, Dict, Optional

# Assuming BaseNode is available at this path as per project instructions
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JSONFormatterNode(BaseNode):
    """
    A Vishustra processing node that formats input data as a JSON string.

    This node can take various types of data (dictionaries, lists, basic types,
    or even existing JSON strings) and output a pretty-printed JSON string.
    If the input data is already a JSON string, it will be parsed and then
    re-serialized with the specified formatting options. Data containing
    non-serializable Python objects will result in a TypeError.

    Configuration via context:
    - 'indent' (Optional[int]): The number of spaces to use for indentation.
                                Defaults to 2. Use None for a compact JSON output.
    - 'sort_keys' (bool): If True, output dictionary keys will be sorted
                          alphabetically. Defaults to False.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "JSONFormatter"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Formats the input data into a JSON string according to context parameters.

        Args:
            data (Any): The input data to be formatted. Can be a Python object
                        (dict, list, int, str, etc.) or a JSON string.
            context (Dict[str, Any]): A dictionary containing runtime context.
                                      Can include 'indent' (int) and 'sort_keys' (bool).

        Returns:
            str: The formatted JSON string.

        Raises:
            TypeError: If the input data contains non-serializable types.
            json.JSONDecodeError: If the input data is a string that was intended
                                  to be JSON but is malformed, and this causes
                                  a processing issue (though typically handled by logging
                                  and attempting direct serialization of the string).
        """
        indent: Optional[int] = context.get('indent', 2)
        sort_keys: bool = context.get('sort_keys', False)

        obj_to_serialize: Any = data

        if isinstance(data, str):
            try:
                # If data is a string, attempt to parse it as JSON first.
                # This allows re-formatting existing JSON strings.
                obj_to_serialize = json.loads(data)
                logger.debug("Input data was a JSON string, parsed successfully for re-formatting.")
            except json.JSONDecodeError as e:
                # If the string is not valid JSON, we log a warning.
                # We then proceed to attempt to serialize the raw string itself.
                # This ensures simple strings like "hello" become "\"hello\"" and
                # malformed JSON strings like "{'key': 'value'}" become "\"{'key': 'value'}\"".
                logger.warning(
                    f"Input string data is not valid JSON. Attempting to serialize it directly. "
                    f"Error: {e}. Data snippet: '{data[:100]}{'...' if len(data) > 100 else ''}'"
                )
                # obj_to_serialize remains the original string 'data' in this case

        try:
            formatted_json = json.dumps(
                obj_to_serialize,
                indent=indent,
                sort_keys=sort_keys
            )
            logger.debug(f"Data formatted to JSON successfully. Indent: {indent}, Sort keys: {sort_keys}.")
            return formatted_json
        except TypeError as e:
            logger.error(
                f"Failed to serialize data to JSON due to a TypeError. "
                f"Data type: {type(obj_to_serialize).__name__}. Error: {e}"
            )
            raise TypeError(
                f"Data contains non-serializable types for JSON: {type(obj_to_serialize).__name__}. {e}"
            ) from e
        except Exception as e:
            # Catch any other unexpected serialization errors
            logger.error(
                f"An unexpected error occurred during JSON serialization. "
                f"Data type: {type(obj_to_serialize).__name__}. Error: {e}"
            )
            raise