
import json
import logging
from typing import Any, Dict, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JSONFormatterNode(BaseNode):
    """
    A Vishustra node that formats data to a strict JSON-representable Python object
    (dictionary or list).

    This node performs the following transformations:
    - Parses JSON strings or bytes into their corresponding Python dictionary or list objects.
    - Validates existing Python dictionaries or lists to ensure they contain only
      JSON-serializable types, raising an error if non-serializable types are found.
      This acts as a normalization step, confirming the data's JSON readiness.
    - Raises appropriate errors for input data that cannot be parsed or represented as valid JSON.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "JSONFormatter"

    def process(self, data: Any, context: Dict[str, Any]) -> Union[Dict[str, Any], list]:
        """
        Processes the input data to ensure it is a valid JSON-representable
        Python object (dictionary or list).

        Args:
            data: The input data, expected to be a JSON string, bytes, dict, or list.
            context: A dictionary of contextual information for processing.
                     Currently not used by this node but available for future extensions.

        Returns:
            A Python dictionary or list representing the JSON data.

        Raises:
            ValueError: If the input data is an invalid JSON string or bytes.
            TypeError: If the input data type is not a string, bytes, dict, or list,
                       or if an existing dict/list contains non-JSON-serializable types.
        """
        if isinstance(data, (str, bytes)):
            try:
                # Attempt to parse JSON string/bytes
                formatted_data = json.loads(data)
                logger.debug(f"[{self.node_name}] Successfully parsed JSON string/bytes.")
                return formatted_data
            except json.JSONDecodeError as e:
                logger.error(
                    f"[{self.node_name}] Failed to decode JSON from input data: {e}. "
                    f"Input sample: {str(data)[:100]}..."
                )
                raise ValueError(f"Input data is not a valid JSON string/bytes: {e}") from e
            except Exception as e:
                # Catch other potential unexpected errors during loads
                logger.error(
                    f"[{self.node_name}] An unexpected error occurred during JSON parsing: {e}. "
                    f"Input type: {type(data)}"
                )
                raise TypeError(f"Could not parse JSON: {e}") from e
        elif isinstance(data, (dict, list)):
            try:
                # For existing dicts/lists, serialize and deserialize to validate
                # and ensure strict JSON-representable types. This step catches
                # non-JSON-serializable types (e.g., datetime objects, sets) embedded in the structure.
                json_string = json.dumps(data)
                formatted_data = json.loads(json_string) # Deserializing back to ensure original object structure
                logger.debug(f"[{self.node_name}] Successfully validated and normalized existing dict/list.")
                return formatted_data
            except TypeError as e:
                logger.error(
                    f"[{self.node_name}] Input dict/list contains non-JSON-serializable types: {e}. "
                    f"Problematic data sample: {str(data)[:100]}..."
                )
                raise TypeError(f"Input dict/list contains non-JSON-serializable types: {e}") from e
            except Exception as e:
                # Catch other potential unexpected errors during validation
                logger.error(
                    f"[{self.node_name}] An unexpected error occurred during dict/list validation: {e}. "
                    f"Input type: {type(data)}"
                )
                raise TypeError(f"Could not validate/normalize dict/list as JSON: {e}") from e
        else:
            logger.error(f"[{self.node_name}] Received unsupported data type: {type(data)}. Expected str, bytes, dict, or list.")
            raise TypeError(
                f"Unsupported data type for JSONFormatterNode. Expected str, bytes, dict, or list, got {type(data)}."
            )

