import json
import logging
from typing import Any, Dict

# Assuming BaseNode is correctly available at this path as per project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JSONFormatterNode(BaseNode):
    """
    A Vishustra processing node that formats input data into a JSON string.
    
    This node takes any serializable Python object and converts it into 
    its JSON string representation. It supports optional pretty-printing
    and key sorting via context parameters.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "JSONFormatter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by converting it into a JSON string.

        The `context` dictionary can contain the following optional keys,
        which directly map to parameters of `json.dumps`:
        - `indent` (int | None): If provided, specifies the number of spaces
          to use for indentation, enabling pretty-printing. Defaults to `None`
          (compact JSON).
        - `sort_keys` (bool): If `True`, the output of dictionaries will be
          sorted by key. Defaults to `False`.

        Args:
            data (Any): The data to be serialized into a JSON string.
                        This can be any Python object that `json.dumps` can handle.
            context (Dict[str, Any]): A dictionary containing additional
                                      parameters for JSON serialization.

        Returns:
            str: A JSON string representation of the input data.

        Raises:
            ValueError: If the input data is not JSON serializable (e.g., contains
                        objects like sets, complex numbers, or custom objects without
                        a custom serializer).
            Exception: For any other unexpected errors during serialization.
        """
        indent = context.get("indent")
        sort_keys = context.get("sort_keys", False)

        try:
            # Attempt to serialize the data to a JSON string
            json_string = json.dumps(data, indent=indent, sort_keys=sort_keys)
            logger.debug(f"Node '{self.node_name}' successfully formatted data to JSON.")
            return json_string
        except TypeError as e:
            # Catch specific error for non-serializable objects
            # Log with details to aid debugging without printing
            logger.error(
                f"Node '{self.node_name}' failed to serialize data to JSON due to "
                f"unserializable type: '{e}'. Data type: {type(data).__name__}. "
                f"Data sample (first 100 chars): '{str(data)[:100]}...'"
            )
            # Re-raise as a ValueError to indicate a clear data processing issue
            raise ValueError(f"Input data is not JSON serializable: {e}") from e
        except Exception as e:
            # Catch any other unexpected errors during serialization
            logger.error(
                f"Node '{self.node_name}' encountered an unexpected error during JSON "
                f"serialization: '{e}'. Data type: {type(data).__name__}. "
                f"Data sample (first 100 chars): '{str(data)[:100]}...'"
            )
            # Re-raise the original exception to propagate the unexpected failure
            raise