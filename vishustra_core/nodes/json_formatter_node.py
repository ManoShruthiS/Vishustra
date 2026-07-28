import json
import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JSONFormatterNode(BaseNode):
    """
    A processing node designed to format input data into a pretty-printed JSON string.
    
    This node aims to standardize data output by converting various input types
    into a human-readable JSON string format (using 2-space indentation).
    
    - If the input `data` is already a string, the node first attempts to parse it
      as JSON. If parsing is successful, it then re-serializes the resulting Python
      object with formatting. If parsing fails (e.g., the string is not valid JSON),
      a warning is logged, and the original string is returned, as it cannot be
      formatted as JSON.
    - If the input `data` is a Python dictionary, list, or another inherently
      JSON-serializable type (like int, float, bool, None), it is directly
      serialized into a formatted JSON string.
    - If the input `data` is a Python object that is not JSON-serializable,
      a warning is logged, and the original object is returned.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "JSON Formatter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, attempting to format it as a pretty-printed
        JSON string.

        Args:
            data: The input data to be formatted. This can be a string, dict, list,
                  or other JSON-serializable type.
            context: A dictionary containing contextual information for the node's
                     operation. (This node does not currently utilize the context
                     but it is part of the BaseNode interface).

        Returns:
            A pretty-printed JSON string if the formatting is successful.
            Returns `None` if the input `data` itself is `None`.
            Returns the original `data` if it cannot be formatted (e.g., it's a
            non-JSON string or a non-serializable Python object), with a
            corresponding warning logged.
        """
        if data is None:
            logger.debug(f"{self.node_name}: Received None data. Returning None.")
            return None

        # Prepare the data for serialization. If it's a string, try to parse it first.
        data_to_serialize = data

        if isinstance(data, str):
            try:
                data_to_serialize = json.loads(data)
                logger.debug(f"{self.node_name}: Successfully parsed input string into a Python object.")
            except json.JSONDecodeError as e:
                logger.warning(
                    f"{self.node_name}: Input string is not valid JSON. Cannot format. "
                    f"Returning original string. Error: {e}"
                )
                return data  # Return original string if it's not valid JSON
            except Exception as e:
                logger.warning(
                    f"{self.node_name}: An unexpected error occurred while parsing the input string. "
                    f"Returning original string. Error: {e}"
                )
                return data

        try:
            # Serialize the (potentially parsed) data into a formatted JSON string
            formatted_json = json.dumps(data_to_serialize, indent=2)
            logger.debug(f"{self.node_name}: Successfully formatted data as JSON.")
            return formatted_json
        except TypeError as e:
            logger.warning(
                f"{self.node_name}: Data (after potential parsing) is not JSON-serializable. "
                f"Returning original data. Error: {e}"
            )
            return data  # Return original data if it cannot be serialized
        except Exception as e:
            # Catch any other unexpected serialization errors
            logger.error(
                f"{self.node_name}: An unhandled error occurred during JSON serialization. "
                f"Returning original data. Error: {e}",
                exc_info=True
            )
            return data
