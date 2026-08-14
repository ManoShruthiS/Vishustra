import json
import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class JSONFormatterNode(BaseNode):
    """
    A Vishustra processing node that formats input data into a JSON string.

    This node ensures consistent JSON output, supporting both Python objects
    and existing JSON strings as input. It can also apply pretty-printing
    based on 'indent' value provided in the context.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "JSONFormatter"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Formats the input data into a JSON string.

        If the input `data` is already a JSON string, it will be parsed and then
        re-serialized to ensure consistent formatting (e.g., applying indentation).
        If `data` is a Python object (dict, list, int, str, etc.), it will be
        directly serialized to a JSON string.

        Args:
            data (Any): The data to be formatted. Can be a Python object or a JSON string.
            context (Dict[str, Any]): A dictionary containing contextual information.
                                      Optional keys:
                                      - 'indent' (int): If provided, the JSON output
                                                        will be pretty-printed with
                                                        the specified indentation level.

        Returns:
            str: A JSON formatted string.

        Raises:
            ValueError: If the input data cannot be serialized to JSON,
                        or if a string input is not valid JSON.
        """
        indent = context.get("indent", None)

        obj_to_serialize = data
        if isinstance(data, str):
            try:
                # If input is a string, assume it might be a JSON string that needs re-formatting
                obj_to_serialize = json.loads(data)
                logger.debug("Input data was a JSON string, successfully parsed for re-formatting.")
            except json.JSONDecodeError as e:
                logger.error(f"JSONFormatterNode failed to parse input string as JSON: {e}. "
                             f"Data start: '{data[:200]}'")
                raise ValueError(
                    f"Input string is not valid JSON and cannot be formatted: {e}"
                ) from e

        try:
            formatted_json = json.dumps(obj_to_serialize, indent=indent)
            logger.debug("Data successfully formatted as JSON by JSONFormatterNode.")
            return formatted_json
        except TypeError as e:
            logger.error(f"JSONFormatterNode failed to serialize data to JSON: {e}. "
                         f"Data type: {type(obj_to_serialize)}. Data: {obj_to_serialize}")
            raise ValueError(
                f"Data of type '{type(obj_to_serialize).__name__}' is not JSON serializable: {e}"
            ) from e