import json
import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JSONFormatterNode(BaseNode):
    """
    A processing node designed to ensure data is a valid and consistently formatted JSON string.

    This node performs a critical role in data pipeline hygiene by:
    1.  **Serializing Python Objects**: If the input `data` is a Python object (e.g., dictionary, list,
        integer, boolean, or None), it attempts to convert it into a compact JSON string.
    2.  **Validating and Normalizing JSON Strings**: If the input `data` is already a string,
        it first attempts to parse it as JSON to validate its correctness. If successful,
        it then re-serializes the parsed object into a compact JSON string. This process
        ensures consistency (e.g., removal of excess whitespace) and confirms the string's validity.
    3.  **Robust Error Handling**: Should any serialization or parsing operation fail,
        a `ValueError` is raised, clearly indicating that the input could not be
        formatted into valid JSON. This prevents malformed data from proceeding further.

    The output of this node is always a compact JSON string, or an error is raised.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "JSONFormatter"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data, ensuring the output is a valid JSON string.

        Args:
            data: The input data to be formatted. This can be any Python object
                  that is either JSON-serializable or a valid JSON string itself.
            context: A dictionary containing contextual information for the processing
                     workflow (not directly used by this specific node, but required
                     by the `BaseNode` interface).

        Returns:
            A compact JSON string representation of the input data. The string is
            guaranteed to be valid JSON.

        Raises:
            ValueError: If the input data cannot be serialized into JSON, or if
                        an input string is found to be malformed (invalid JSON).
        """
        formatted_json_string: str

        try:
            if isinstance(data, str):
                logger.debug("Input data is a string. Attempting to parse and re-serialize as JSON.")
                # Attempt to parse the string to validate it and then re-serialize for consistency
                parsed_data = json.loads(data)
                formatted_json_string = json.dumps(parsed_data, ensure_ascii=False, separators=(',', ':'))
                logger.debug("Successfully parsed and re-serialized input JSON string.")
            else:
                logger.debug("Input data is a Python object. Attempting to serialize to JSON.")
                # Directly serialize Python objects to a compact JSON string
                formatted_json_string = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
                logger.debug("Successfully serialized Python object to JSON string.")
        except json.JSONDecodeError as e:
            # Handles cases where input `data` is a string but not valid JSON
            logger.error(f"Failed to decode input string as valid JSON: {e}", exc_info=True)
            raise ValueError(f"Input string is not valid JSON: {e}") from e
        except TypeError as e:
            # Handles cases where input `data` is a Python object that cannot be serialized to JSON
            logger.error(f"Failed to serialize data to JSON due to a type error: {e}", exc_info=True)
            raise ValueError(f"Data is not JSON serializable: {e}") from e
        except Exception as e:
            # Catch any other unexpected exceptions during the process
            logger.error(f"An unexpected error occurred during JSON formatting: {e}", exc_info=True)
            raise ValueError(f"An unexpected error prevented JSON formatting: {e}") from e

        return formatted_json_string