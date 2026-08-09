
import json
import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JSONFormatterNode(BaseNode):
    """
    A Vishustra node that takes arbitrary data (or a JSON string)
    and formats it as a JSON string.

    This node is useful for ensuring consistent JSON output format,
    especially for pretty-printing for readability or normalizing
    JSON structures from various sources.

    The node attempts to parse string input as JSON first. If the input
    data is not a valid JSON string or a JSON-serializable Python object,
    it will raise an appropriate error.

    Context Parameters:
    - 'json_indent' (int | None, optional): The indentation level for the JSON
      output. If an integer >= 0, it specifies the number of spaces for indentation.
      If `None`, no indentation is applied (compact output). Defaults to 2 spaces
      if not provided or an invalid value is given.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the JSON Formatter node."""
        return "JSON Formatter Node"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Formats the input data as a JSON string.

        The method handles various input types:
        - If `data` is a JSON-serializable Python object (dict, list, int, str, etc.),
          it will be directly serialized.
        - If `data` is a string, the node will first attempt to parse it as JSON.
          If successful, the parsed object will then be re-serialized to apply
          the desired formatting. If parsing fails, a ValueError is raised.

        Args:
            data (Any): The data to be formatted. Can be a Python object
                        (dict, list, str, int, etc.) or a JSON string.
            context (Dict[str, Any]): A dictionary containing contextual
                                       information. Can include 'json_indent'.

        Returns:
            str: A formatted JSON string.

        Raises:
            ValueError: If the input data is a string but not valid JSON.
            TypeError: If the input data is not JSON serializable.
        """
        logger.debug("JSONFormatterNode: Processing initiated.")

        indent_level = context.get('json_indent')

        if indent_level is None:
            # User explicitly set None or didn't provide, means compact output for json.dumps
            logger.debug("JSONFormatterNode: 'json_indent' is None, using compact output.")
        elif isinstance(indent_level, int) and indent_level >= 0:
            logger.debug(f"JSONFormatterNode: Using 'json_indent' level: {indent_level}")
            # indent_level is valid
        else:
            # Invalid indent value, default to 2 and warn
            logger.warning(
                f"JSONFormatterNode: Invalid 'json_indent' value '{indent_level}' "
                "in context. Expected int >= 0 or None. Defaulting to 2 spaces."
            )
            indent_level = 2

        obj_to_serialize = data

        # If the input is a string, try to parse it first to ensure it's valid JSON
        if isinstance(data, str):
            try:
                obj_to_serialize = json.loads(data)
                logger.debug("JSONFormatterNode: Input data was a JSON string, successfully parsed.")
            except json.JSONDecodeError as e:
                logger.error(
                    f"JSONFormatterNode: Input data is a string but not valid JSON. Error: {e}"
                )
                raise ValueError(f"Input data is not valid JSON: {e}") from e

        # Now, serialize the (potentially parsed) Python object to a JSON string
        try:
            formatted_json_string = json.dumps(obj_to_serialize, indent=indent_level)
            logger.debug("JSONFormatterNode: Data successfully serialized to JSON string.")
            return formatted_json_string
        except TypeError as e:
            logger.error(f"JSONFormatterNode: Data cannot be serialized to JSON: {e}")
            raise TypeError(f"Data cannot be serialized to JSON: {e}") from e

