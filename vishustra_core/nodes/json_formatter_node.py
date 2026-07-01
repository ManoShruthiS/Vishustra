import logging
import json
from typing import Any, Dict, Union

# BaseNode is expected to be in this path within the Vishustra framework
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JsonFormatterNode(BaseNode):
    """
    A processing node designed to robustly format input data into a JSON-compatible
    Python object (dictionary or list).

    This node handles various input types:
    - If the input is already a Python dictionary or list, it is returned as-is.
    - If the input is a string, it attempts to parse it as JSON.
    - For other Python types, it attempts to serialize them to a JSON string
      and then deserialize to ensure JSON validity and structure.

    The primary goal is to produce a Python `dict` or `list` representation of JSON.
    If the input data, after processing, results in a scalar JSON value (e.g., number,
    boolean, null, or a simple string that is not JSON), this node will log a warning
    and return an empty dictionary (`{}`) to indicate that no structured JSON object
    or array could be formed.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "JsonFormatterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Union[Dict[str, Any], list]:
        """
        Processes the input data, attempting to convert it into a JSON-compatible
        Python dictionary or list.

        Args:
            data: The input data to be formatted. Can be a string, dict, list,
                  or other JSON-serializable type.
            context: A dictionary containing contextual information for the processing.
                     This node does not currently utilize the context, but it is
                     passed through as per the `BaseNode` interface.

        Returns:
            A Python dictionary or list representing the formatted JSON structure.
            Returns an empty dictionary (`{}`) if the processed data is a scalar
            JSON value (e.g., `None`, `5`, `"hello"`).

        Raises:
            ValueError: If a string input cannot be parsed as valid JSON.
            TypeError: If the input data is not JSON-serializable.
            RuntimeError: For unexpected errors during the JSON processing chain.
        """
        if isinstance(data, (dict, list)):
            logger.debug(f"[{self.node_name}] Input data is already a dict or list; returning as-is.")
            return data
        
        processed_data: Any
        if isinstance(data, str):
            try:
                processed_data = json.loads(data)
                logger.debug(f"[{self.node_name}] Successfully parsed string input as JSON.")
            except json.JSONDecodeError as e:
                logger.error(
                    f"[{self.node_name}] Failed to parse string input as JSON. "
                    f"Error: {e}. Data snippet: '{data[:200]}{'...' if len(data) > 200 else ''}'"
                )
                raise ValueError(f"Input string is not valid JSON.") from e
        else:
            # For other types, first attempt to serialize to a JSON string,
            # then deserialize to ensure it becomes a proper Python object.
            try:
                json_string = json.dumps(data)
                processed_data = json.loads(json_string)
                logger.debug(f"[{self.node_name}] Successfully serialized and deserialized non-string input.")
            except TypeError as e:
                logger.error(
                    f"[{self.node_name}] Input data of type {type(data).__name__} is not JSON serializable. "
                    f"Error: {e}. Data: '{data}'"
                )
                raise TypeError(f"Input data is not JSON serializable.") from e
            except json.JSONDecodeError as e:
                # This is an unlikely scenario if json.dumps succeeded, but good for robustness.
                logger.error(
                    f"[{self.node_name}] Unexpected error: data serialized but failed to deserialize. "
                    f"Error: {e}. Data: '{data}'"
                )
                raise RuntimeError(f"Unexpected error during JSON serialization/deserialization.") from e

        # Ensure the final output is a dictionary or a list, as typically expected from a
        # "JSON formatter" node in an orchestration framework dealing with structured outputs.
        if isinstance(processed_data, (dict, list)):
            return processed_data
        else:
            logger.warning(
                f"[{self.node_name}] Formatted data resulted in a scalar value "
                f"(type: {type(processed_data).__name__}). Returning an empty dict "
                f"as no structured object or array could be formed. Original data: '{data}'"
            )
            return {}
