import json
import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JSONFormatterNode(BaseNode):
    """
    A Vishustra processing node that formats input data into a JSON string.

    This node attempts to serialize the input data using `json.dumps()`.
    It can optionally take a 'json_indent' value from the context dictionary
    to pretty-print the output JSON.

    If the input data is not JSON serializable, a ValueError is raised.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "JSONFormatter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, attempting to convert it into a JSON string.

        Args:
            data: The input data to be formatted. Can be any Python object
                  that is JSON serializable.
            context: A dictionary containing contextual information.
                     Can include "json_indent" (int) for pretty-printing.

        Returns:
            A string representing the JSON formatted data.

        Raises:
            ValueError: If the input data is not JSON serializable.
            RuntimeError: For unexpected errors during the JSON serialization process.
        """
        logger.debug(f"JSONFormatterNode '{self.node_name}' received data of type: {type(data)}")

        # Retrieve optional 'json_indent' from context for pretty-printing
        indent = context.get("json_indent", None)
        if indent is not None and not isinstance(indent, int):
            logger.warning(
                f"Context 'json_indent' was provided but not an integer ({type(indent)}). "
                "Ignoring indent and proceeding without pretty-printing."
            )
            indent = None

        try:
            # Attempt to serialize the data to a JSON string
            formatted_json_string = json.dumps(data, indent=indent)
            logger.info(f"Data successfully formatted as JSON string by '{self.node_name}'.")
            return formatted_json_string
        except TypeError as e:
            error_msg = (
                f"Failed to format data as JSON in '{self.node_name}'. "
                f"Data of type {type(data)} is not JSON serializable. Error: {e}"
            )
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg) from e
        except Exception as e:
            # Catch any other unexpected errors during serialization
            error_msg = f"An unexpected error occurred during JSON formatting in '{self.node_name}': {e}"
            logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
