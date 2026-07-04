import json
import logging
from typing import Any, Dict

# Simulating the import path as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JSONFormatterNode(BaseNode):
    """
    A Vishustra node that formats input data into a standardized JSON string.

    This node aims to serialize any input data into a valid JSON string.
    If the input data is already a string, it first attempts to parse it as JSON
    to ensure consistent re-formatting (e.g., applying indentation). If the input
    string is not valid JSON, it will be treated as a literal string and serialized
    as such (e.g., the string "hello" becomes the JSON string '"hello"').

    Configuration options can be passed via the context dictionary:
    - 'indent': An integer value to specify the indentation level for pretty-printing
                the JSON output. Defaults to None, resulting in compact JSON.
                Example: `{'indent': 2}` for 2-space indentation.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "JSONFormatter"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data, formatting it into a JSON string.

        Args:
            data: The input data to be formatted. This can be any JSON-serializable
                  Python object (dict, list, str, int, float, bool, None), or
                  a string which may or may not be valid JSON.
            context: A dictionary containing operational context.
                     Supports an optional 'indent' key for JSON pretty-printing.

        Returns:
            A string representing the JSON-formatted data.

        Raises:
            ValueError: If the input data, or a component within it, is not
                        JSON-serializable according to Python's `json` module rules.
        """
        indent_level = context.get('indent', None)
        if not (indent_level is None or (isinstance(indent_level, int) and indent_level >= 0)):
            logger.warning(
                f"[{self.node_name}] Invalid 'indent' value '{indent_level}' provided in context. "
                "Expected None or a non-negative integer. Falling back to default (None - compact JSON)."
            )
            indent_level = None

        data_to_serialize = data

        # If the input is a string, attempt to parse it first.
        # This allows re-formatting of existing JSON strings and proper handling.
        if isinstance(data, str):
            try:
                data_to_serialize = json.loads(data)
                logger.debug(f"[{self.node_name}] Successfully parsed input string as JSON object.")
            except json.JSONDecodeError as e:
                # If it's a string but not valid JSON, we proceed with the original string.
                # This means it will be serialized as a literal string in JSON (e.g., '"my string"').
                logger.warning(
                    f"[{self.node_name}] Input string could not be parsed as valid JSON. "
                    f"Serializing as a JSON literal string. Error: {e}"
                )
            except Exception as e:
                # Catch any unexpected errors during string parsing.
                logger.error(
                    f"[{self.node_name}] An unexpected error occurred during JSON string parsing: {e}. "
                    "Proceeding to serialize original string."
                )

        try:
            # Serialize the (potentially parsed) Python object into a JSON string.
            # ensure_ascii=False allows direct output of non-ASCII characters without escaping.
            formatted_json = json.dumps(data_to_serialize, indent=indent_level, ensure_ascii=False)
            logger.debug(f"[{self.node_name}] Data successfully formatted as JSON.")
            return formatted_json
        except TypeError as e:
            # This error occurs if `data_to_serialize` contains types that `json` cannot handle,
            # such as Python objects without a `__dict__` or a custom serialization method.
            logger.error(
                f"[{self.node_name}] Input data is not JSON-serializable: {e}. "
                f"Problematic type: {type(data_to_serialize)}"
            )
            raise ValueError(f"Input data is not JSON-serializable: {e}")
        except Exception as e:
            # Catch any other unforeseen issues during the serialization process.
            logger.exception(f"[{self.node_name}] An unexpected error occurred during JSON formatting.")
            raise
