import json
import logging
from typing import Any, Dict, Optional

# Assuming this path relative to the project root where Vishustra is installed
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JSONFormatterNode(BaseNode):
    """
    A Vishustra processing node that serializes and formats Python data structures
    (dictionaries, lists, primitives) into a pretty-printed JSON string.

    This node is useful for ensuring structured output, enhancing readability
    of logs, or preparing data for APIs that expect formatted JSON.
    """

    def __init__(self, indent: Optional[int] = 2, ensure_ascii: bool = False):
        """
        Initializes the JSONFormatterNode.

        Args:
            indent (Optional[int]): The number of spaces to use for indentation.
                                    If None, JSON will be compact. Defaults to 2.
            ensure_ascii (bool): If true, ensure all non-ASCII characters in the
                                 output are escaped. If false, these characters
                                 are output directly. Defaults to False.
        """
        self._indent = indent
        self._ensure_ascii = ensure_ascii
        logger.debug(
            f"{self.node_name} initialized with indent={indent}, ensure_ascii={ensure_ascii}"
        )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "JSONFormatterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data, converting it into a formatted JSON string.

        Args:
            data (Any): The Python data structure to be serialized (e.g., dict, list, str, int, float, bool, None).
                        If `data` is already a JSON string, it will be treated as a Python string
                        and `json.dumps` will attempt to serialize it directly. This might
                        not be the intended behavior if re-formatting an *existing* JSON string
                        (e.g., changing its indentation) is desired; for that, consider parsing
                        the string first using a `JSONParserNode`.
            context (Dict[str, Any]): A dictionary containing contextual information
                                      for the processing pipeline. Not directly used
                                      for transformation in this node, but available.

        Returns:
            str: The formatted JSON string representation of the input data.

        Raises:
            ValueError: If the input data cannot be serialized into a valid JSON string.
        """
        try:
            formatted_json = json.dumps(
                data,
                indent=self._indent,
                ensure_ascii=self._ensure_ascii,
            )
            logger.info(f"Successfully formatted data to JSON string for node '{self.node_name}'")
            return formatted_json
        except TypeError as e:
            error_msg = (
                f"Failed to serialize data into JSON for node '{self.node_name}'. "
                f"Input data of type '{type(data).__name__}' is not JSON serializable. Error: {e}"
            )
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg) from e
        except Exception as e:
            # Catching any other unexpected issues during serialization
            error_msg = (
                f"An unexpected error occurred during JSON serialization for node '{self.node_name}'. "
                f"Error: {e}"
            )
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg) from e