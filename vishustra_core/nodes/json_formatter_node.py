import json
import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class JsonFormatterNode(BaseNode):
    """
    A processing node that takes any serializable data and formats it into
    a JSON string.

    This node can also attempt to re-format an existing JSON string if
    provided as input, allowing for consistent indentation or re-validation.

    Context parameters:
        - `json_indent` (int, optional): The indentation level for the JSON
          output. If not provided, the output will be a compact JSON string.
        - `ensure_ascii` (bool, optional): If `True` (default), all non-ASCII
          characters in the output are escaped. If `False`, they are output
          directly.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "JsonFormatter"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data, attempting to serialize it into a JSON string.

        Args:
            data (Any): The input data to be formatted. This can be a Python
                        dictionary, list, or any other JSON-serializable object.
                        If it's a string, the node will first attempt to parse
                        it as JSON to re-format it.
            context (Dict[str, Any]): A dictionary containing additional
                                      processing parameters, such as
                                      `json_indent` and `ensure_ascii`.

        Returns:
            str: The JSON formatted string.

        Raises:
            TypeError: If the input `data` is not JSON-serializable.
            json.JSONDecodeError: If the input `data` is a string but not
                                  valid JSON, and strict parsing is desired.
                                  (Currently, it logs and attempts to serialize
                                  the string directly).
        """
        obj_to_serialize = data
        original_data_is_string = False

        if isinstance(data, str):
            original_data_is_string = True
            try:
                # Attempt to parse as JSON if it's already a string,
                # so we can re-format it properly.
                obj_to_serialize = json.loads(data)
                logger.debug("Input data was a JSON string, parsed for re-formatting.")
            except json.JSONDecodeError as e:
                # If it's a string but not valid JSON, we'll try to serialize
                # the string itself later.
                logger.warning(
                    f"Input string data is not valid JSON, treating as raw string for serialization: {e}"
                )
            except Exception as e:
                # Catch other potential errors during parsing
                logger.error(
                    f"Unexpected error while attempting to parse input string as JSON: {e}"
                )

        indent = context.get("json_indent")
        ensure_ascii = context.get("ensure_ascii", True)

        try:
            formatted_json = json.dumps(
                obj_to_serialize, indent=indent, ensure_ascii=ensure_ascii
            )
            logger.info("Data successfully formatted into JSON string.")
            return formatted_json
        except TypeError as e:
            if original_data_is_string and obj_to_serialize == data:
                # This means the original string was not valid JSON and also not directly serializable as a string literal.
                logger.error(
                    f"Cannot serialize non-JSON string '{data[:100]}...' into valid JSON: {e}"
                )
            else:
                logger.error(f"Input data is not JSON-serializable: {e}")
            raise  # Re-raise to indicate a processing failure
        except Exception as e:
            logger.error(f"An unexpected error occurred during JSON formatting: {e}")
            raise