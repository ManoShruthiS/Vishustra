import json
import logging
from typing import Any, Dict

# Assuming BaseNode is available at this path as per project context
from vishustra_core.nodes.base_node import BaseNode 

logger = logging.getLogger(__name__)

class JsonFormatterNode(BaseNode):
    """
    A processing node designed to ensure input data is consistently represented
    as a properly formatted JSON string.

    This node attempts to serialize Python dictionaries or lists directly into
    JSON. If the input is a string, it first tries to parse it as JSON and then
    re-serializes it for consistent formatting. For other data types, it
    attempts direct JSON serialization.

    In cases where the data cannot be converted to valid JSON, the node logs
    a warning or error and returns the original data, ensuring robustness
    in data pipelines.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "JSON Formatter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, attempting to serialize it into a
        formatted JSON string.

        Configuration options can be passed via the `context` dictionary:
        - `json_formatter_indent` (int | None): An integer specifying the
          indentation level for pretty-printing. A value of `None` will result
          in the most compact JSON representation. Defaults to `4` for readability.

        Args:
            data: The input data, which can be a string, dict, list, or any
                  other serializable Python object.
            context: A dictionary containing contextual information and
                     configuration parameters for the node.

        Returns:
            A formatted JSON string if the processing is successful. If the
            data cannot be converted to valid JSON, the original `data`
            is returned, and an appropriate log message is generated.
        """
        # Retrieve indentation level from context, defaulting to 4 for pretty-printing
        indent = context.get('json_formatter_indent', 4)

        try:
            if isinstance(data, (dict, list)):
                # Data is already a Python object (dict/list), serialize it directly
                formatted_json = json.dumps(data, indent=indent)
                logger.debug("JsonFormatterNode: Successfully formatted Python object to JSON.")
                return formatted_json
            elif isinstance(data, str):
                # Data is a string; attempt to parse it first to ensure validity
                # and then re-serialize for consistent formatting.
                try:
                    parsed_data = json.loads(data)
                    formatted_json = json.dumps(parsed_data, indent=indent)
                    logger.debug("JsonFormatterNode: Successfully parsed and re-formatted JSON string.")
                    return formatted_json
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"JsonFormatterNode: Input string is not valid JSON. "
                        f"Returning original data. Error: {e}"
                    )
                    return data
            else:
                # Data is of another type (e.g., int, float, bool, None).
                # Attempt direct serialization. 'indent' will generally be ignored for scalar types.
                try:
                    formatted_json = json.dumps(data, indent=indent)
                    logger.debug(f"JsonFormatterNode: Successfully formatted scalar data type '{type(data).__name__}' to JSON.")
                    return formatted_json
                except TypeError as e:
                    logger.warning(
                        f"JsonFormatterNode: Data of type '{type(data).__name__}' cannot be "
                        f"directly serialized to JSON. Returning original data. Error: {e}"
                    )
                    return data
        except Exception as e:
            # Catch any unexpected errors during the overall JSON processing
            logger.error(
                f"JsonFormatterNode: An unexpected error occurred during JSON formatting. "
                f"Returning original data. Error: {e}",
                exc_info=True  # Include stack trace for debugging unexpected errors
            )
            return data