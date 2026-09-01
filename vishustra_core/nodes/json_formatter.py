import json
import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JsonFormatterNode(BaseNode):
    """
    A processing node that standardizes data into a formatted JSON string.

    This node takes various forms of input (JSON string, Python dict/list,
    or other basic types) and ensures the output is a valid JSON string.
    It can optionally pretty-print the output based on context settings.

    Context Parameters:
    - 'pretty_print' (bool, optional): If True, the output JSON will be
      formatted with indentation. Defaults to False.
    - 'indent_spaces' (int, optional): Number of spaces to use for indentation
      if 'pretty_print' is True. Defaults to 2.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "JSON Formatter"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data, ensuring it is converted to a formatted
        JSON string.

        If the input `data` is a JSON string, it will first be parsed and
        then re-serialized to ensure consistent formatting.
        If `data` is a Python object, it will be directly serialized.

        Args:
            data (Any): The input data to be formatted. Can be a JSON string,
                        a Python dictionary, list, or other serializable types.
            context (Dict[str, Any]): A dictionary containing runtime context
                                      and configuration for the node.

        Returns:
            str: The formatted JSON string representation of the input data.

        Raises:
            json.JSONDecodeError: If the input `data` is a string but not
                                  valid JSON.
            TypeError: If the input `data` contains elements that are not
                       JSON serializable.
        """
        intermediate_data: Any = data
        indent: int | None = None

        if context.get('pretty_print', False):
            indent = context.get('indent_spaces', 2)

        # If data is already a string, attempt to parse it first to ensure
        # it's valid and to allow re-serialization with new formatting.
        if isinstance(data, str):
            try:
                intermediate_data = json.loads(data)
                logger.debug("Successfully parsed input string as JSON.")
            except json.JSONDecodeError as e:
                logger.error(f"Input string is not valid JSON: {e}")
                raise e # Re-raise to signal a critical processing failure
            except Exception as e:
                logger.error(f"Unexpected error while parsing JSON string: {e}")
                raise e

        # Now, `intermediate_data` is a Python object. Serialize it to a string.
        try:
            formatted_json_string = json.dumps(intermediate_data, indent=indent)
            logger.debug("Successfully formatted data into JSON string.")
            return formatted_json_string
        except TypeError as e:
            logger.error(f"Data contains non-serializable types: {e}")
            raise e # Re-raise to signal a critical processing failure
        except Exception as e:
            logger.error(f"Unexpected error during JSON serialization: {e}")
            raise e
