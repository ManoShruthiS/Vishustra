import json
import logging
from typing import Any, Dict, Optional

# Assuming BaseNode is available at this path within the Vishustra framework
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JsonFormatterNode(BaseNode):
    """
    A Vishustra processing node that serializes input data into a JSON string.

    This node is crucial for standardizing data into a JSON format, which can be
    required for subsequent processing steps, API calls, or for final structured
    output. It supports pretty-printing through an optional 'indent' parameter
    during initialization, allowing for human-readable output when desired.
    """

    def __init__(self, indent: Optional[int] = None):
        """
        Initializes the JsonFormatterNode.

        Args:
            indent: An integer representing the number of spaces to use for
                    indentation when pretty-printing the JSON output. If `None`,
                    the output will be a compact JSON string with no extra whitespace.
        """
        self._indent = indent
        logger.debug(f"JsonFormatterNode initialized with indent setting: {indent}")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "JsonFormatter"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data by serializing it into a JSON string.

        Args:
            data: The input data to be serialized. This data must be
                  JSON-serializable (e.g., Python dictionaries, lists, strings,
                  numbers, booleans, or None).
            context: A dictionary containing contextual information for the current
                     orchestration run. This node does not explicitly utilize the
                     context, but it is provided as part of the BaseNode interface.

        Returns:
            A string representing the JSON-serialized form of the input data.

        Raises:
            TypeError: If the input `data` is not JSON-serializable (e.g., contains
                       unsupported Python types like sets or custom objects without
                       a `__json__` method or custom encoder).
            RuntimeError: For any other unexpected errors encountered during the
                          JSON serialization process.
        """
        try:
            formatted_json = json.dumps(data, indent=self._indent)
            logger.info(f"Node '{self.node_name}' successfully formatted data to JSON. "
                        f"Output length: {len(formatted_json)} characters.")
            return formatted_json
        except TypeError as e:
            error_msg = (
                f"Node '{self.node_name}' failed to format data to JSON. "
                f"Data is not JSON serializable. Type: '{type(data).__name__}'. Error: {e}"
            )
            logger.error(error_msg, exc_info=True)
            raise TypeError(error_msg) from e
        except Exception as e:
            # Catching general exceptions as a safeguard for unforeseen issues with json.dumps
            error_msg = (
                f"Node '{self.node_name}' encountered an unexpected error during JSON formatting. "
                f"Error: {e}"
            )
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e