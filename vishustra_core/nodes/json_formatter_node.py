import json
import logging
from typing import Any, Dict, Optional

# Assuming this path is correct based on the project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JSONFormatterNode(BaseNode):
    """
    A Vishustra processing node that serializes input data into a JSON string.

    This node provides flexible JSON formatting, allowing control over indentation
    and ASCII-encoding via constructor parameters or runtime context overrides.
    """

    def __init__(self, indent: Optional[int] = None, ensure_ascii: bool = True):
        """
        Initializes the JSONFormatterNode with default serialization parameters.

        Args:
            indent (Optional[int]): If provided, the JSON output will be pretty-printed
                                    with this indent level. A value of None results in
                                    the most compact representation. Defaults to None.
            ensure_ascii (bool): If True, all non-ASCII characters in the output
                                 are escaped with \\uXXXX sequences. If False,
                                 these characters are output directly. Defaults to True.
        """
        self._default_indent = indent
        self._default_ensure_ascii = ensure_ascii
        logger.debug(
            f"Initialized JSONFormatterNode with default indent={indent}, "
            f"ensure_ascii={ensure_ascii}"
        )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "JSONFormatter"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data, attempting to serialize it into a JSON string.

        Configuration parameters (`indent`, `ensure_ascii`) can be dynamically
        overridden by values provided in the `context` dictionary under the
        keys 'json_indent' and 'json_ensure_ascii' respectively. This allows
        for fine-grained control during pipeline execution.

        Args:
            data (Any): The input data to be serialized. This can be any Python
                        object that `json.dumps` can handle (e.g., dict, list, str,
                        int, float, bool, None).
            context (Dict[str, Any]): A dictionary containing runtime context
                                     variables. May contain 'json_indent' (int)
                                     or 'json_ensure_ascii' (bool) to override
                                     the node's default settings.

        Returns:
            str: The JSON-formatted string representation of the input data.

        Raises:
            ValueError: If the input data is not JSON serializable. This typically
                        occurs for complex custom objects without a defined
                        serialization method.
            RuntimeError: For any unexpected errors during the serialization process.
        """
        # Determine effective parameters, prioritizing context over defaults
        effective_indent = context.get("json_indent", self._default_indent)
        effective_ensure_ascii = context.get("json_ensure_ascii", self._default_ensure_ascii)

        logger.info(
            f"[{self.node_name}] Attempting to format data to JSON. "
            f"Effective Indent: {effective_indent}, "
            f"Effective Ensure ASCII: {effective_ensure_ascii}"
        )

        try:
            json_string = json.dumps(
                data,
                indent=effective_indent,
                ensure_ascii=effective_ensure_ascii
            )
            logger.debug(f"[{self.node_name}] Successfully formatted data to JSON.")
            return json_string
        except TypeError as e:
            error_msg = (
                f"[{self.node_name}] Failed to serialize data to JSON. "
                f"Input data of type '{type(data).__name__}' is not JSON serializable. "
                f"Error details: {e}"
            )
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg) from e
        except Exception as e:
            # Catching a broader exception for any other unforeseen issues during serialization
            error_msg = (
                f"[{self.node_name}] An unexpected error occurred during JSON serialization. "
                f"Input data type: '{type(data).__name__}'. Error details: {e}"
            )
            logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e