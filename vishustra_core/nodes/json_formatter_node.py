import json
import logging
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JSONFormatterNode(BaseNode):
    """
    A Vishustra processing node that formats input data into a JSON string.

    This node takes any JSON-serializable Python object and converts it
    into a JSON string, optionally pretty-printing it with a specified indent.
    """

    def __init__(self, indent: Optional[int] = None):
        """
        Initializes the JSONFormatterNode.

        Args:
            indent: If provided, the JSON output will be pretty-printed with
                    this indentation level. If None, the output will be compact.
        """
        self._indent = indent
        logger.debug(f"JSONFormatterNode initialized with indent={indent}")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "JSONFormatter"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data by converting it into a JSON string.

        Args:
            data: The input Python object (e.g., dict, list, string, int, bool)
                  to be serialized to JSON.
            context: A dictionary containing contextual information for the node.
                     Not directly used by this node but required by the BaseNode interface.

        Returns:
            A string representing the JSON-formatted data.

        Raises:
            ValueError: If the input data is not JSON serializable.
            RuntimeError: For unexpected errors during JSON serialization.
        """
        try:
            json_string = json.dumps(data, indent=self._indent)
            logger.debug(f"Successfully formatted data to JSON. Output length: {len(json_string)}")
            return json_string
        except TypeError as e:
            logger.error(
                f"Failed to serialize data to JSON due to TypeError: {e}. "
                f"Input data type: {type(data)}. Data sample: {str(data)[:200]}",
                exc_info=True
            )
            raise ValueError(f"Input data is not JSON serializable: {e}") from e
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during JSON serialization: {e}. "
                f"Input data type: {type(data)}. Data sample: {str(data)[:200]}",
                exc_info=True
            )
            raise RuntimeError(f"Unexpected error during JSON serialization: {e}") from e