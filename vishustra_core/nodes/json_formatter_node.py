import json
import logging
from typing import Any, Dict, Optional, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JsonFormatterNode(BaseNode):
    """
    A processing node that formats input data into a JSON string.

    It attempts to parse string/bytes data as JSON first. If successful, or if the input
    is already a Python dictionary or list, it then serializes it into a JSON string.
    Non-JSON string inputs will be treated as a string literal and serialized accordingly.
    Unserializable Python objects will result in an error and the original data being returned.
    """

    def __init__(self, indent: Optional[int] = None):
        """
        Initializes the JsonFormatterNode.

        Args:
            indent (Optional[int]): The indentation level for the JSON output.
                                    If None (default), the JSON will be compact.
                                    If a non-negative integer, arrays and objects
                                    will be pretty-printed with that indent level.
        """
        self._indent = indent
        logger.debug(f"JsonFormatterNode initialized with indent={indent}")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "JsonFormatter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, converting it into a formatted JSON string.

        Args:
            data (Any): The input data to be formatted. Can be a string, bytes,
                        dictionary, list, or other serializable Python object.
            context (Dict[str, Any]): A dictionary containing contextual information
                                      for the processing. Not directly used by this node.

        Returns:
            Any: A JSON formatted string if successful, otherwise the original
                 data if serialization fails.
        """
        payload_to_serialize = data

        if isinstance(data, (str, bytes)):
            try:
                # Attempt to parse string/bytes as JSON first
                payload_to_serialize = json.loads(data)
                logger.debug("Input data successfully parsed as JSON string/bytes.")
            except json.JSONDecodeError:
                # If it's not a valid JSON string, treat the string itself as the payload
                logger.warning(
                    f"Input data of type {type(data)} could not be decoded as JSON. "
                    "Treating input as a literal for serialization. Data sample: "
                    f"{str(data)[:100]}{'...' if len(str(data)) > 100 else ''}"
                )
            except Exception as e:
                logger.error(
                    f"Unexpected error while decoding input data of type {type(data)}: {e}"
                )
                # Fallback to original data for serialization attempt
                payload_to_serialize = data
        
        try:
            # Serialize the (potentially parsed) payload to a JSON string
            formatted_json_string = json.dumps(payload_to_serialize, indent=self._indent)
            logger.debug("Data successfully formatted into JSON string.")
            return formatted_json_string
        except TypeError as e:
            logger.error(
                f"Failed to serialize data of type {type(payload_to_serialize)} to JSON. "
                f"Likely contains unserializable objects: {e}. "
                "Returning original data."
            )
            return data
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during JSON serialization: {e}. "
                "Returning original data."
            )
            return data