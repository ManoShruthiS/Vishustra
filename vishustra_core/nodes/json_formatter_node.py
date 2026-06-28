import json
import logging
from typing import Any, Dict

# Assuming BaseNode is correctly available at this path within the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JSONFormatterNode(BaseNode):
    """
    A Vishustra node designed to serialize input data into a JSON string,
    applying optional formatting.

    This node handles various input types:
    - If the input `data` is a Python object (e.g., dict, list, int, bool),
      it is directly serialized to a JSON string.
    - If the input `data` is a string, the node first attempts to parse it
      as a JSON string. If successful, the parsed Python object is then
      re-serialized. This ensures consistent formatting (e.g., indentation)
      even for already-JSON-like string inputs.
    - If the input `data` is a string but not valid JSON, it will be
      serialized as a literal JSON string (e.g., "hello" becomes '"hello"').

    Configuration for `json.dumps()` can be provided via the `context`
    dictionary under the key `'json_dumps_options'`.
    Example context for pretty-printing: `{'json_dumps_options': {'indent': 2}}`
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "JSONFormatter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, converting it into a formatted JSON string.

        Args:
            data: The input data to be serialized. This can be any Python object
                  that is JSON serializable, or a string which may or may not be
                  a JSON string itself.
            context: A dictionary containing additional information or configuration.
                     It can include `'json_dumps_options'` as a dictionary of
                     keyword arguments to be passed to `json.dumps()`.

        Returns:
            A string representing the JSON formatted data.

        Raises:
            TypeError: If the input data (or its parsed form) is not JSON serializable.
            Exception: For any other unexpected errors during the serialization process.
        """
        # Extract json.dumps options from the context, defaulting to an empty dict
        json_dumps_options = context.get('json_dumps_options', {})

        # Determine the actual Python object to serialize
        data_to_serialize = data
        if isinstance(data, str):
            try:
                # Attempt to parse the string as JSON. If successful, use the parsed object.
                parsed_data = json.loads(data)
                data_to_serialize = parsed_data
                logger.debug(f"Node '{self.node_name}': Input data was a JSON string, successfully parsed for re-serialization.")
            except json.JSONDecodeError:
                # If the string is not valid JSON, log and proceed with the original string.
                logger.debug(f"Node '{self.node_name}': Input data is a string but not valid JSON. Serializing as a literal string.")
            except Exception as e:
                # Catch any other unexpected errors during parsing of string input.
                logger.warning(
                    f"Node '{self.node_name}': Unexpected error during JSON parsing of string input: {e}. "
                    "Proceeding with original string.",
                    exc_info=True
                )
                data_to_serialize = data # Fallback to original data if parsing failed unexpectedly

        try:
            # Serialize the determined data_to_serialize into a JSON string
            formatted_json_string = json.dumps(data_to_serialize, **json_dumps_options)
            logger.info(f"Node '{self.node_name}': Data successfully formatted as JSON.")
            return formatted_json_string
        except TypeError as e:
            # Handle cases where the data cannot be serialized to JSON
            logger.error(
                f"Node '{self.node_name}': Data of type '{type(data_to_serialize).__name__}' "
                f"is not JSON serializable. Error: {e}",
                exc_info=True
            )
            raise TypeError(f"Data is not JSON serializable: {e}") from e
        except Exception as e:
            # Catch any other unexpected exceptions during the serialization process
            logger.error(
                f"Node '{self.node_name}': An unexpected error occurred during JSON serialization. "
                f"Error: {e}",
                exc_info=True
            )
            raise # Re-raise unknown exceptions to propagate the failure