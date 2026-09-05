import json
import logging
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JsonFormatterNode(BaseNode):
    """
    A Vishustra processing node that serializes input data into a JSON string.

    This node takes any Python object and attempts to convert it into a JSON
    formatted string. It supports configurable indentation and key sorting
    via the `context` dictionary.

    Configuration parameters can be provided in the `context` dictionary:
    - `json_indent` (int | None, optional): The indentation level for pretty-printing.
      Defaults to 2. Set to `None` for the most compact output (no newlines or extra spaces).
    - `json_sort_keys` (bool, optional): If `True`, output dictionary keys will be
      sorted alphabetically. Defaults to `False`.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "JsonFormatterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data by serializing it into a JSON string.

        Args:
            data (Any): The input data to be serialized. This can be any
                        Python object that is JSON-serializable (e.g., dict, list,
                        string, number, boolean, None).
            context (Dict[str, Any]): A dictionary containing contextual information
                                       and configuration parameters for this node.
                                       Expected keys for JSON formatting:
                                       - `json_indent`: Integer or None for pretty-printing indent.
                                       - `json_sort_keys`: Boolean to sort keys.

        Returns:
            str: The JSON-formatted string representation of the input data.

        Raises:
            ValueError: If the input data is not JSON-serializable.
            RuntimeError: For any unexpected errors during serialization.
        """
        # Retrieve configuration from context, providing sensible defaults
        indent: Optional[int] = context.get("json_indent", 2)
        sort_keys: bool = context.get("json_sort_keys", False)

        logger.debug(
            "[%s] Attempting to format data to JSON with indent=%s, sort_keys=%s. Input data type: %s",
            self.node_name, indent, sort_keys, type(data)
        )

        try:
            json_string = json.dumps(data, indent=indent, sort_keys=sort_keys)
            logger.info("[%s] Successfully formatted data to JSON.", self.node_name)
            return json_string
        except TypeError as e:
            logger.error(
                "[%s] Failed to serialize data to JSON due to non-serializable type: %s. "
                "Data type: %s. Data sample: %.100s",
                self.node_name, e, type(data), str(data)
            )
            raise ValueError(f"Data is not JSON-serializable: {e}") from e
        except Exception as e:
            # Catch any other unexpected errors during serialization
            logger.critical(
                "[%s] An unexpected error occurred during JSON serialization: %s. Input data type: %s",
                self.node_name, e, type(data)
            )
            raise RuntimeError(
                f"An unexpected error occurred in {self.node_name}: {e}"
            ) from e
