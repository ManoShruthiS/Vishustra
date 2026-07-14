import json
import logging
from typing import Any, Dict

# Assuming BaseNode is correctly imported from its path in the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JSONFormatterNode(BaseNode):
    """
    A Vishustra processing node that formats input data into a JSON string.

    This node intelligently handles various input types for serialization:
    - If the input `data` is a Python dictionary or list, it will be directly serialized.
    - If the input `data` is an existing JSON string, it will be parsed and then
      re-serialized to ensure consistent formatting (e.g., applying indentation).
    - Other JSON-serializable primitive types (int, float, bool, None) will be
      serialized directly as their JSON equivalents.
    - If `data` is a string that is not valid JSON, it will be treated as a
      string value and serialized as such (e.g., "hello" becomes '"hello"').

    Configuration for `json.dumps` can be provided via the `context` dictionary:
    - `json_indent`: An integer specifying the indentation level for pretty-printing.
                     If not provided or not an integer, no indentation will be used.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "JSONFormatter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Formats the input data into a JSON string, applying specified indentation if any.

        Args:
            data: The input data to be formatted. This can be any JSON-serializable
                  type, including a pre-existing JSON string or a Python object.
            context: A dictionary containing contextual information.
                     Expected keys include 'json_indent' (int) for pretty-printing.

        Returns:
            A JSON-formatted string representing the input data.

        Raises:
            ValueError: If the input data is not JSON-serializable (e.g., contains
                        unsupported Python objects like sets or custom classes).
            RuntimeError: For unexpected errors during the serialization process.
        """
        logger.debug(f"JSONFormatterNode '{self.node_name}' received data for processing.")

        object_to_serialize: Any = data

        # Determine indentation level from context
        indent = context.get('json_indent', None)
        if indent is not None and not isinstance(indent, int):
            logger.warning(
                f"Context parameter 'json_indent' expected an integer for node "
                f"'{self.node_name}', but received type '{type(indent).__name__}'. "
                f"Falling back to default (no indentation)."
            )
            indent = None
        elif isinstance(indent, int) and indent < 0:
            logger.warning(
                f"Context parameter 'json_indent' for node '{self.node_name}' "
                f"received a negative integer ({indent}). Falling back to default (no indentation)."
            )
            indent = None

        if isinstance(data, str):
            try:
                # Attempt to parse the string as JSON. If successful, the parsed
                # object will be re-serialized, allowing for re-formatting (e.g., indentation).
                object_to_serialize = json.loads(data)
                logger.debug(
                    f"Input string was successfully parsed as JSON by node '{self.node_name}'."
                )
            except json.JSONDecodeError:
                # If the string is not valid JSON, treat it as a literal string value
                # to be serialized (e.g., "hello world" -> '"hello world"').
                logger.debug(
                    f"Input string could not be parsed as JSON by node '{self.node_name}'. "
                    f"Attempting to serialize the string itself as a value."
                )
                object_to_serialize = data # Keep the original string
            except Exception as e:
                # Catch any other unexpected errors during string parsing.
                logger.error(
                    f"Unexpected error when attempting to parse input string as JSON in node "
                    f"'{self.node_name}': {type(e).__name__}: {e}. "
                    f"Proceeding to serialize the original string data."
                )
                object_to_serialize = data

        try:
            # Serialize the determined Python object into a JSON string.
            formatted_json_string = json.dumps(object_to_serialize, indent=indent)
            logger.info(f"Data successfully formatted to JSON string by node '{self.node_name}'.")
            return formatted_json_string
        except TypeError as e:
            # This error occurs if the object_to_serialize contains non-JSON-serializable types.
            logger.error(
                f"Input data to node '{self.node_name}' is not JSON serializable: "
                f"{type(e).__name__}: {e}. Data type: {type(object_to_serialize).__name__}"
            )
            raise ValueError(
                f"JSONFormatterNode '{self.node_name}' received non-serializable data. "
                f"Error: {e}. Data type: {type(object_to_serialize).__name__}"
            ) from e
        except Exception as e:
            # Catch any other unexpected serialization errors.
            logger.error(
                f"An unexpected error occurred during JSON serialization in node "
                f"'{self.node_name}': {type(e).__name__}: {e}"
            )
            raise RuntimeError(
                f"JSONFormatterNode '{self.node_name}' failed due to an unexpected "
                f"serialization error: {e}"
            ) from e
