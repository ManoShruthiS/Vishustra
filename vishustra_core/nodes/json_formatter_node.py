import json
import logging
from typing import Any, Dict

# Assuming BaseNode is located at this path within the Vishustra project structure.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JSONFormatterNode(BaseNode):
    """
    A Vishustra node that formats input data as a pretty-printed JSON string.

    This node is designed to standardize the format of JSON data, making it
    more readable and consistent. It handles various input types:
    - If the input `data` is a string, it attempts to parse it as JSON. If successful,
      it then pretty-prints the parsed JSON. If parsing fails, the original string
      is returned, and a warning is logged.
    - If the input `data` is a Python dictionary, list, or other basic JSON-serializable
      type (int, float, bool, None), it directly serializes and pretty-prints it.
    - For other data types, it attempts generic JSON serialization. If this fails
      (e.g., due to non-serializable objects), the original data is returned,
      and an error is logged.

    Configuration options for JSON serialization (like `indent` level and `sort_keys`)
    can be passed via the `context` dictionary under the 'json_formatter_config' key.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "JSON Formatter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, attempting to format it as a pretty-printed
        JSON string.

        Args:
            data: The input data, which can be a JSON string, a Python dict/list,
                  or other serializable types.
            context: A dictionary containing contextual information. This can include
                     optional configuration for JSON serialization under the
                     'json_formatter_config' key. For example:
                     `{'json_formatter_config': {'indent': 4, 'sort_keys': False}}`

        Returns:
            A pretty-printed JSON string if the data can be successfully formatted.
            Otherwise, the original `data` is returned, and a warning or error
            is logged to indicate the issue.
        """
        config = context.get('json_formatter_config', {})
        indent = config.get('indent', 2)
        sort_keys = config.get('sort_keys', True)

        logger.debug(
            f"[{self.node_name}] Starting process for data type: {type(data).__name__}. "
            f"Config: indent={indent}, sort_keys={sort_keys}"
        )

        try:
            if isinstance(data, str):
                try:
                    # Attempt to parse the string as JSON
                    parsed_data = json.loads(data)
                    logger.debug(f"[{self.node_name}] Successfully parsed string data as JSON.")
                    # Then re-serialize with desired formatting
                    return json.dumps(parsed_data, indent=indent, sort_keys=sort_keys)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"[{self.node_name}] Input string is not valid JSON. "
                        f"Returning original string. Error: {e}"
                    )
                    return data
            elif isinstance(data, (dict, list, int, float, bool)) or data is None:
                # Directly serialize Python objects that are intrinsically JSON-serializable
                logger.debug(f"[{self.node_name}] Input is a JSON-serializable Python object.")
                return json.dumps(data, indent=indent, sort_keys=sort_keys)
            else:
                # Attempt to serialize other types. This might fail if objects are not serializable.
                logger.warning(
                    f"[{self.node_name}] Unexpected data type '{type(data).__name__}' for direct JSON formatting. "
                    f"Attempting generic serialization, but this might fail. "
                    f"Consider converting data to a dict or list first."
                )
                return json.dumps(data, indent=indent, sort_keys=sort_keys)

        except TypeError as e:
            logger.error(
                f"[{self.node_name}] Data of type '{type(data).__name__}' is not JSON serializable. "
                f"Returning original data. Error: {e}"
            )
            return data
        except Exception as e:
            # Catch any other unforeseen exceptions during the process
            logger.exception(
                f"[{self.node_name}] An unexpected error occurred during JSON formatting. "
                f"Returning original data. Error: {e}"
            )
            return data