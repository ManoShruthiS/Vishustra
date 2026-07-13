import json
import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JSONFormatterNode(BaseNode):
    """
    A Vishustra node that serializes input data into a JSON string.

    This node takes any serializable Python object and converts it into a
    formatted JSON string. It supports optional indentation and key sorting
    via configuration provided in the processing context.

    Configuration can be passed via the `context` dictionary under the key
    `'json_formatter_options'`. This nested dictionary can contain:
    - `'indent'`: (int | None) If provided, specifies the indentation
                  level for pretty-printing. Defaults to 2 spaces.
    - `'sort_keys'`: (bool) If True, output dictionary keys will be
                     sorted alphabetically. Defaults to False.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "JSONFormatter"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Serializes the input data into a JSON string.

        The formatting behavior (indentation, key sorting) can be controlled
        through the 'json_formatter_options' key in the `context` dictionary.

        Args:
            data: The input data to be formatted as JSON. This can be any
                  Python object that is JSON-serializable (e.g., dict, list,
                  string, int, float, bool, None).
            context: A dictionary containing contextual information and
                     optional configuration for JSON serialization.
                     Example context:
                     `{"json_formatter_options": {"indent": 4, "sort_keys": True}}`

        Returns:
            A string representing the JSON-formatted data.

        Raises:
            TypeError: If the input `data` is not JSON-serializable.
            RuntimeError: For any unexpected errors during serialization that are
                          not related to data type.
        """
        options = context.get("json_formatter_options", {})
        # Default to 2-space indentation for readability, None for compact output
        indent = options.get("indent", 2)
        sort_keys = options.get("sort_keys", False)

        logger.debug(
            f"Attempting to format data to JSON with indent={indent}, sort_keys={sort_keys}. "
            f"Input data type: {type(data)}"
        )

        try:
            formatted_json_string = json.dumps(data, indent=indent, sort_keys=sort_keys)
            logger.info("Successfully formatted data to JSON.")
            return formatted_json_string
        except TypeError as e:
            logger.error(
                f"JSONFormatterNode failed: Input data is not JSON-serializable. "
                f"Data type: {type(data)}. Error: {e}",
                exc_info=True
            )
            # Re-raise as TypeError, as it accurately describes the problem with the input data.
            raise TypeError(f"Input data of type {type(data)} is not JSON-serializable: {e}") from e
        except Exception as e:
            logger.critical(
                f"An unexpected critical error occurred during JSON formatting in {self.node_name}. Error: {e}",
                exc_info=True
            )
            # Catch any other unexpected exceptions and wrap them in a RuntimeError
            raise RuntimeError(f"JSONFormatterNode encountered an unexpected error during processing: {e}") from e