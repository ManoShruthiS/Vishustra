import json
import logging
from typing import Any, Dict

# Assuming BaseNode is available at this path as per project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JSONFormatterNode(BaseNode):
    """
    A Vishustra processing node that formats Python objects or JSON strings
    into a standardized, optionally pretty-printed, JSON string representation.

    This node is designed to normalize data into a consistent JSON string format,
    handling both raw Python objects and existing JSON strings as input.
    It provides configuration for indentation and key sorting.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "JSONFormatter"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Formats the input data into a JSON string, applying specified formatting options.

        The node first attempts to interpret the input `data`:
        - If `data` is a string, it tries to parse it as JSON. If successful,
          the parsed Python object is used. If parsing fails, the original
          string itself is treated as the object to be serialized (e.g.,
          the string "hello" would be serialized to '"hello"').
        - If `data` is not a string, it's directly treated as a Python object
          to be serialized.

        Args:
            data: The input data, which can be any Python object (dict, list, str,
                  int, etc.) or a string potentially containing JSON.
            context: A dictionary containing additional processing parameters.
                     Recognized keys:
                     - 'indent': (int | None, optional) The number of spaces to use
                                 for indentation. Defaults to 4 for pretty-printing.
                                 Use `None` for the most compact JSON output.
                     - 'sort_keys': (bool, optional) If True, output dictionary
                                    keys will be sorted alphabetically. Defaults to False.

        Returns:
            A formatted JSON string.

        Raises:
            TypeError: If the final Python object intended for serialization
                       is not JSON serializable (e.g., a custom class instance
                       without a `__json__` method or similar serializer).
            Exception: Catches and re-raises any other unexpected errors during
                       the JSON processing lifecycle.
        """
        obj_to_serialize = data
        
        # Extract formatting parameters from the context, providing sensible defaults
        indent = context.get('indent', 4)  # Default to 4 spaces for pretty-printing
        sort_keys = context.get('sort_keys', False)

        logger.debug(
            f"[{self.node_name}] Preparing to format data. "
            f"Indent: {indent}, Sort Keys: {sort_keys}."
        )

        if isinstance(data, str):
            try:
                # Attempt to parse the input string as JSON
                parsed_data = json.loads(data)
                obj_to_serialize = parsed_data
                logger.debug(f"[{self.node_name}] Successfully parsed input string as a JSON object.")
            except json.JSONDecodeError:
                # If the string is not valid JSON, we'll serialize the string itself.
                # E.g., json.dumps("just a string") results in '"just a string"'.
                logger.warning(
                    f"[{self.node_name}] Input string is not valid JSON. "
                    "Proceeding to serialize the string content directly."
                )
            except Exception as e:
                logger.error(
                    f"[{self.node_name}] An unexpected error occurred while attempting "
                    f"to parse input string as JSON: {e}", exc_info=True
                )
                raise # Re-raise unexpected parsing errors

        try:
            # Serialize the (potentially parsed) Python object into a JSON string
            formatted_json_string = json.dumps(
                obj_to_serialize,
                indent=indent,
                sort_keys=sort_keys
            )
            logger.info(f"[{self.node_name}] Data successfully formatted into JSON string.")
            return formatted_json_string
        except TypeError as e:
            # This error occurs if `obj_to_serialize` contains non-JSON-serializable types
            logger.error(
                f"[{self.node_name}] Failed to serialize data of type "
                f"'{type(obj_to_serialize).__name__}' to JSON: {e}", exc_info=True
            )
            raise TypeError(
                f"Input data (or its parsed form) is not JSON serializable: {e}"
            ) from e
        except Exception as e:
            # Catch any other unforeseen issues during serialization
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during JSON serialization: {e}", exc_info=True
            )
            raise # Re-raise for upstream handling