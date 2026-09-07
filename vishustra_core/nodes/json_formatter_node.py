import json
import logging
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JSONFormatterNode(BaseNode):
    """
    A Vishustra processing node that serializes input data into a JSON string.

    This node takes any Python object that is JSON serializable and converts it
    into a JSON formatted string. It provides comprehensive control over the
    serialization process through parameters that mirror `json.dumps`.
    """

    def __init__(self, indent: Optional[int] = None, ensure_ascii: bool = True, **kwargs):
        """
        Initializes the JSONFormatterNode.

        Args:
            indent (Optional[int]): If provided, JSON output will be pretty-printed
                                    with the specified indent level. If None,
                                    output will be compact.
            ensure_ascii (bool): If True (default), all non-ASCII characters in
                                 the output are escaped. If False, non-ASCII
                                 characters are output directly.
            **kwargs: Additional keyword arguments to pass directly to `json.dumps`,
                      such as `sort_keys`, `separators`, `default`, etc.
        """
        self._dumps_kwargs = {'indent': indent, 'ensure_ascii': ensure_ascii}
        self._dumps_kwargs.update(kwargs)
        logger.debug(f"JSONFormatterNode initialized with json.dumps parameters: {self._dumps_kwargs}")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "JSONFormatter"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data, serializing it into a JSON string.

        This method attempts to convert the provided `data` into a JSON string
        using the `json.dumps` function. If the data is not JSON serializable,
        a `TypeError` is raised.

        Args:
            data (Any): The input data to be formatted as a JSON string. This can be
                        any Python object that is JSON serializable (e.g., dict, list,
                        str, int, float, bool, None).
            context (Dict[str, Any]): A dictionary containing contextual information.
                                      This node does not typically use `context` for
                                      its core serialization logic but it's available.

        Returns:
            str: The JSON formatted string representation of the input data.

        Raises:
            TypeError: If the input data contains objects that are not JSON serializable.
            RuntimeError: For any other unexpected errors during the serialization process.
        """
        logger.debug(f"JSONFormatterNode received data of type '{type(data)}' for processing.")

        try:
            # Directly use json.dumps on the input data.
            # This handles various types correctly:
            # - Dicts/Lists become standard JSON objects/arrays.
            # - Primitives (int, float, bool, None) become their JSON equivalents.
            # - Strings become quoted JSON strings (e.g., "hello" becomes ""hello"").
            json_string = json.dumps(data, **self._dumps_kwargs)
            logger.info(f"Successfully formatted data as JSON. Output (first 100 chars): {json_string[:100]}{'...' if len(json_string) > 100 else ''}")
            return json_string
        except TypeError as e:
            logger.error(f"Failed to serialize data of type '{type(data)}' to JSON due to non-serializable content. Error: {e}", exc_info=True)
            # Re-raise as a TypeError, indicating a problem with the input data's type or content.
            raise TypeError(f"Data of type '{type(data)}' is not JSON serializable: {e}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during JSON serialization for data of type '{type(data)}'. Error: {e}", exc_info=True)
            # Catch any other unexpected exceptions during serialization.
            raise RuntimeError(f"An unexpected error occurred during JSON serialization: {e}") from e