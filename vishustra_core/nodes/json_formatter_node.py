import json
import logging
from typing import Any, Dict

# Assuming this import path exists within the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JSONFormatterNode(BaseNode):
    """
    A Vishustra node that serializes input data into a JSON string.

    This node ensures that its output is a valid JSON string.
    If the input data is already a string, it first attempts to parse it as JSON.
    If parsing is successful, the parsed Python object is then re-serialized
    to ensure consistent formatting.
    If parsing fails (meaning the string is not a valid JSON document, e.g., "hello"),
    the string itself is treated as the value to be serialized (e.g., "hello" becomes '"hello"').
    If the data is a Python object (e.g., dict, list, int), it is directly serialized.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "JSONFormatterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by attempting to serialize it into a JSON string.

        Args:
            data: The input data to be formatted. Can be any serializable Python object.
            context: A dictionary containing contextual information for processing.
                     (Not directly used by this node but required by the BaseNode interface).

        Returns:
            A JSON string representation of the input data if successful.
            Returns None if the data cannot be serialized into JSON,
            logging an error in such cases.
        """
        logger.debug(f"[{self.node_name}] Received data for JSON formatting. Type: {type(data)}")

        data_to_serialize = data

        if isinstance(data, str):
            try:
                # Attempt to parse existing string as JSON.
                # If successful, use the parsed object for re-serialization.
                # This ensures consistent formatting (e.g., whitespace, key order)
                # and validates the input if it's already a JSON string.
                parsed_data = json.loads(data)
                data_to_serialize = parsed_data
                logger.debug(f"[{self.node_name}] Input string was a valid JSON document, parsed successfully.")
            except json.JSONDecodeError:
                # If the string is not a valid JSON document, treat the string itself
                # as the value to be serialized. For example, a Python string 'hello'
                # will be serialized to a JSON string '"hello"'.
                logger.debug(
                    f"[{self.node_name}] Input string is not a valid JSON document. "
                    "Treating the string itself as the value to be serialized."
                )
                # data_to_serialize remains `data` which is the original string.
                # json.dumps will correctly serialize a Python string into a JSON string.

        try:
            # Serialize the data to a JSON string.
            # Using `separators=(',', ':')` for compact JSON output,
            # which is often preferred in backend systems for efficiency unless
            # pretty-printing is explicitly required.
            formatted_json = json.dumps(data_to_serialize, separators=(',', ':'))
            logger.info(f"[{self.node_name}] Successfully formatted data as JSON.")
            return formatted_json
        except TypeError as e:
            logger.error(
                f"[{self.node_name}] Failed to serialize data into JSON due to a TypeError: {e}. "
                "The input data type is not JSON serializable or contains non-serializable elements."
            )
            return None
        except Exception as e:
            # Catch any other unexpected serialization errors.
            # `exc_info=True` includes traceback in the log for better debugging.
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during JSON serialization: {e}",
                exc_info=True
            )
            return None