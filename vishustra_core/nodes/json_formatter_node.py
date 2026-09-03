import json
import logging
from typing import Any, Dict

# Assuming BaseNode will be available via this import path in the Vishustra framework
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class JSONFormatterNode(BaseNode):
    """
    A Vishustra processing node designed to format input data into a
    JSON-compatible Python object (dictionary or list).

    This node primarily handles two scenarios:
    1. Parsing a raw JSON string into its corresponding Python object.
    2. Passing through an already existing Python dictionary or list,
       effectively validating it as JSON-compatible.
    """

    def __init__(self):
        """
        Initializes the JSONFormatterNode.
        No specific configuration parameters are required for its core functionality.
        """
        logger.debug("JSONFormatterNode initialized.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "JSONFormatterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, attempting to convert it into a JSON-compatible
        Python object.

        - If `data` is a string, it attempts to parse it as JSON.
        - If `data` is already a dictionary or list, it is returned as is,
          as it's already in a JSON-compatible Python format.
        - For any other data types, a `TypeError` is raised, as they cannot be
          directly formatted into a JSON object by this node without further
          serialization instructions.

        Args:
            data: The input data. Expected to be either a JSON string, a Python
                  dictionary, or a Python list.
            context: A dictionary containing contextual information relevant to the
                     current processing pipeline. This node does not modify or
                     directly use the context for its formatting logic.

        Returns:
            The parsed Python object (a dictionary or a list) if `data` was a
            JSON string, or the original dictionary/list if it was already one.

        Raises:
            ValueError: If `data` is a string but fails JSON parsing due to invalid syntax.
            TypeError: If `data` is neither a string, dictionary, nor a list,
                       indicating an unsupported input type for this formatter.
        """
        logger.info(f"[{self.node_name}] Starting process for input data of type: {type(data).__name__}")

        if isinstance(data, str):
            try:
                processed_data = json.loads(data)
                logger.debug(f"[{self.node_name}] Successfully parsed JSON string into Python object.")
                return processed_data
            except json.JSONDecodeError as e:
                logger.error(f"[{self.node_name}] Failed to parse JSON string: {e}. "
                             f"Input data snippet: '{data[:200]}...'")
                raise ValueError(f"Input string is not valid JSON: {e}") from e
        elif isinstance(data, (dict, list)):
            logger.debug(f"[{self.node_name}] Data is already a dictionary or list; returning as is.")
            return data
        else:
            logger.error(f"[{self.node_name}] Unsupported data type for JSON formatting: {type(data).__name__}. "
                         f"Expected str, dict, or list.")
            raise TypeError(
                f"Unsupported data type for {self.node_name}. Expected 'str', 'dict', or 'list', "
                f"but received '{type(data).__name__}'."
            )