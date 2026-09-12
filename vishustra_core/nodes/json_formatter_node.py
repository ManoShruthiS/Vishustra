
import json
import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JSONFormatterNode(BaseNode):
    """
    A processing node designed to serialize input data into a JSON string.

    This node intelligently handles various input types, attempting to parse
    existing JSON strings for re-formatting (e.g., pretty-printing) and
    directly serializing Python objects. It provides robust error handling
    for non-JSON-serializable data and invalid context parameters.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "JSONFormatter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Converts the input data into a formatted JSON string.

        The `context` dictionary can optionally specify an 'indent' level
        for pretty-printing the JSON output.

        Args:
            data: The input data to be formatted. This can be any JSON-serializable
                  Python object (e.g., dict, list, str, int, float, bool, None).
                  If `data` is already a string, the node attempts to parse it
                  as JSON first. If successful, it will be re-serialized. If not,
                  the string itself will be treated as the value to be serialized.
            context: A dictionary containing contextual information.
                     Expected keys:
                     - 'json_indent' (optional, int): The number of spaces to use
                       for indentation to pretty-print the JSON output. If not
                       provided or invalid, no indentation will be applied.

        Returns:
            A string representing the JSON-formatted version of the input data.

        Raises:
            TypeError: If the input `data` or any of its components are not
                       JSON-serializable (e.g., Python objects like sets, functions,
                       or custom classes without a `__json__` method).
            ValueError: Propagated from `json.dumps` if underlying serialization
                        encounters an issue not covered by TypeError.
        """
        indent_level = context.get('json_indent')

        if indent_level is not None:
            if not isinstance(indent_level, int) or indent_level < 0:
                logger.warning(
                    f"[{self.node_name}] Invalid 'json_indent' value '{indent_level}' "
                    "in context. Expected a non-negative integer. Proceeding without indentation."
                )
                indent_level = None

        data_to_serialize = data
        if isinstance(data, str):
            try:
                # Attempt to parse existing JSON strings to allow re-formatting
                # and ensure consistency if 'indent' is applied.
                data_to_serialize = json.loads(data)
            except json.JSONDecodeError:
                # If it's a string but not valid JSON, treat the string itself
                # as the value to be serialized (e.g., "plain text" -> "\"plain text\"").
                logger.debug(
                    f"[{self.node_name}] Input string could not be decoded as JSON. "
                    "Treating as a literal string for serialization."
                )
            except Exception as e:
                logger.error(
                    f"[{self.node_name}] Unexpected error parsing string data: {e}",
                    exc_info=True
                )
                raise

        try:
            formatted_json_string = json.dumps(data_to_serialize, indent=indent_level)
            logger.debug(
                f"[{self.node_name}] Successfully formatted data to JSON. "
                f"Output length: {len(formatted_json_string)}"
            )
            return formatted_json_string
        except TypeError as e:
            logger.error(
                f"[{self.node_name}] Failed to serialize data of type '{type(data)}' "
                f"to JSON due to non-serializable object: {e}"
            )
            raise
        except Exception as e:
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during JSON formatting: {e}",
                exc_info=True
            )
            raise

