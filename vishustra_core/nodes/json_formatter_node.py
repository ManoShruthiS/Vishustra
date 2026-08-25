import json
import logging
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JSONFormatterNode(BaseNode):
    """
    A Vishustra node that formats input data into a JSON string.

    This node can take structured Python data (dict, list, etc.) or a JSON string
    and convert it into a consistently formatted JSON string. It supports
    pretty-printing via an `indent` parameter.
    """

    def __init__(self, indent: Optional[int] = None):
        """
        Initializes the JSONFormatterNode.

        Args:
            indent (Optional[int]): If provided, the JSON output will be
                                    pretty-printed with the specified indent level.
                                    If None (default), the JSON will be compact.
        """
        self._indent = indent
        logger.debug(f"JSONFormatterNode initialized with indent={self._indent}")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "JSONFormatter"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data by converting it into a JSON string.

        If the input `data` is already a string, it attempts to parse it as JSON
        first to ensure validity before re-formatting. If `data` is a Python
        object, it directly serializes it to JSON.

        Args:
            data (Any): The input data to be formatted. This can be a Python
                        object (dict, list, int, bool, str) or a JSON string.
            context (Dict[str, Any]): The context dictionary for the current operation.
                                      (Not directly used by this node, but required by BaseNode interface).

        Returns:
            str: A JSON string representation of the input data.

        Raises:
            ValueError: If the input `data` is a string but not valid JSON,
                        or if the input `data` is not JSON-serializable.
            RuntimeError: For any unexpected errors during processing.
        """
        logger.info(f"JSONFormatterNode processing data of type: {type(data)}")

        try:
            intermediate_data: Any
            if isinstance(data, str):
                # If data is a string, try to parse it first to ensure it's valid JSON
                intermediate_data = json.loads(data)
                logger.debug("Input data was a string, successfully parsed as JSON.")
            else:
                # If not a string, assume it's a Python object suitable for serialization
                intermediate_data = data
                logger.debug("Input data is not a string, proceeding to serialize.")

            # Format the (parsed or original Python) data into a JSON string
            formatted_json = json.dumps(intermediate_data, indent=self._indent)
            logger.debug("Data successfully formatted into JSON string.")
            return formatted_json
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse input string as valid JSON: {e}. Input start: '{str(data)[:200]}'")
            raise ValueError(f"Input data is not valid JSON: {e}") from e
        except TypeError as e:
            logger.error(f"Input data is not JSON serializable: {e}. Data type: {type(data)}")
            raise ValueError(f"Input data is not JSON serializable: {e}") from e
        except Exception as e:
            # Catch any other unexpected errors during processing
            logger.critical(f"An unexpected error occurred during JSON formatting: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error during JSON formatting: {e}") from e