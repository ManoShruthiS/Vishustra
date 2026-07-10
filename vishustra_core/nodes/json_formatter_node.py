
import json
import logging
from typing import Any, Dict

# Assuming BaseNode is correctly importable from this path in the Vishustra project
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JsonFormatterNode(BaseNode):
    """
    A Vishustra processing node that formats input data into a consistently
    structured and human-readable JSON string.

    This node takes arbitrary Python data, attempts to serialize it into
    a JSON string, and applies formatting options (like indentation)
    specified in the context. If the input data is already a JSON string,
    it attempts to parse and then re-serialize it to ensure consistent
    formatting across the orchestration.

    Context parameters for JSON serialization (optional):
    - 'json_indent' (int | None): The number of spaces to use for indentation.
                                  If None (default), the JSON will be compact.
                                  Example: 2 or 4.
    - 'json_ensure_ascii' (bool): If True (default), ensure all non-ASCII
                                  characters are escaped. If False,
                                  these characters are output directly.
    - 'json_sort_keys' (bool): If True, output of dictionaries will be
                               sorted by key. Defaults to False.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "JsonFormatter"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Formats the input data into a JSON string, applying context-defined options.

        If the input `data` is a string, the node first attempts to parse it as
        JSON. If successful, it re-serializes the parsed data. If the string
        is not valid JSON, a ValueError is raised.

        Args:
            data (Any): The input data to be formatted. This can be a Python
                        dictionary, list, string (potentially a JSON string),
                        or any other JSON-serializable type.
            context (Dict[str, Any]): A dictionary containing runtime information
                                       and configuration parameters for JSON
                                       serialization (e.g., 'json_indent').

        Returns:
            str: A formatted JSON string representation of the input data.

        Raises:
            ValueError: If the input data is a string but cannot be parsed as
                        valid JSON.
            TypeError: If the input data (or its contents after parsing if a string)
                       is not JSON serializable.
            RuntimeError: For unexpected errors during processing.
        """
        # Retrieve JSON serialization options from the context
        indent = context.get('json_indent')
        ensure_ascii = context.get('json_ensure_ascii', True)
        sort_keys = context.get('json_sort_keys', False)

        processable_data = data

        if isinstance(data, str):
            try:
                # Attempt to parse the string to ensure it's valid JSON
                # This also allows re-formatting of existing JSON strings
                processable_data = json.loads(data)
                logger.debug(
                    f"Input data was identified as a JSON string and parsed for re-formatting."
                )
            except json.JSONDecodeError as e:
                error_msg = (
                    f"Input string data is not valid JSON and cannot be formatted. "
                    f"Details: {e}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg) from e
            except Exception as e:
                error_msg = (
                    f"An unexpected error occurred while parsing input string as JSON. "
                    f"Details: {type(e).__name__}: {e}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e

        try:
            formatted_json_string = json.dumps(
                processable_data,
                indent=indent,
                ensure_ascii=ensure_ascii,
                sort_keys=sort_keys,
            )
            logger.info(
                f"Successfully formatted data as JSON. "
                f"Indent: {indent}, Ensure ASCII: {ensure_ascii}, Sort keys: {sort_keys}."
            )
            return formatted_json_string
        except TypeError as e:
            error_msg = (
                f"Data of type '{type(processable_data).__name__}' or its elements "
                f"are not JSON serializable. Details: {e}"
            )
            logger.error(error_msg)
            raise TypeError(error_msg) from e
        except Exception as e:
            error_msg = (
                f"An unexpected error occurred during JSON serialization. "
                f"Details: {type(e).__name__}: {e}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

