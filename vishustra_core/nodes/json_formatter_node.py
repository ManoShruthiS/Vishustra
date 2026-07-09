import json
import logging
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JSONFormatterNode(BaseNode):
    """
    A Vishustra node that formats input data into a JSON string.

    This node handles various input types:
    - If the input is a Python object (e.g., dict, list, str, int, float, bool, None),
      it will be serialized directly to a JSON string.
    - If the input is already a string, it first attempts to parse it as JSON.
      If successful, it then re-serializes it to ensure consistent formatting
      according to the `indent` configuration.
      If parsing fails (i.e., the string is not valid JSON), an error is raised.

    Configuration:
    - `indent`: Optional integer specifying the number of spaces for pretty-printing JSON.
      If `None` (default), the JSON output will be compact.
    """

    def __init__(self, indent: Optional[int] = None):
        """
        Initializes the JSONFormatterNode.

        Args:
            indent (Optional[int]): The number of spaces to use for indentation
                                    when pretty-printing JSON. If `None`, output
                                    will be compact.
        """
        self._indent = indent
        logger.debug(f"JSONFormatterNode initialized with indent={self._indent}")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "JSONFormatter"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Formats the input data into a JSON string.

        Args:
            data (Any): The input data to be formatted. This can be any Python object
                        that is JSON-serializable, or a string that is a valid JSON document.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing. This node does not directly
                                       use the context, but it is part of the `BaseNode`
                                       interface.

        Returns:
            str: The JSON-formatted string.

        Raises:
            ValueError: If the input data is a string but not valid JSON,
                        or if an unexpected serialization error occurs.
            TypeError: If the input data contains objects that are not
                       JSON-serializable (e.g., a function, a complex object
                       without a custom serializer).
        """
        resolved_data = data

        if isinstance(data, str):
            try:
                # If the input is a string, attempt to parse it first to ensure validity.
                # This also allows re-formatting already-JSON strings.
                resolved_data = json.loads(data)
                logger.debug("Input data was a JSON string, successfully parsed for re-formatting.")
            except json.JSONDecodeError as e:
                error_msg = (
                    f"Input string cannot be parsed as valid JSON. "
                    f"It cannot be formatted: {e}"
                )
                logger.error(error_msg, exc_info=True)
                raise ValueError(error_msg) from e
            except TypeError as e:
                # This should be highly unlikely for `json.loads` on `str`,
                # but included for extreme robustness.
                error_msg = f"Unexpected TypeError when attempting to parse input string: {e}"
                logger.error(error_msg, exc_info=True)
                raise ValueError(error_msg) from e

        try:
            # Serialize the resolved data (either original object or parsed JSON)
            formatted_json = json.dumps(resolved_data, indent=self._indent)
            logger.info("Data successfully formatted as JSON.")
            return formatted_json
        except TypeError as e:
            error_msg = f"Input data contains non-JSON-serializable types: {e}"
            logger.error(error_msg, exc_info=True)
            raise TypeError(error_msg) from e
        except Exception as e:
            # Catch any other unexpected serialization errors
            error_msg = f"An unexpected error occurred during JSON serialization: {e}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg) from e