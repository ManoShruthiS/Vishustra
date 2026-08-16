import json
import logging
from typing import Any, Dict, Optional

# Assuming BaseNode resides in vishustra_core.nodes.base_node as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JSONFormatterNode(BaseNode):
    """
    A Vishustra processing node that formats input data into a consistent JSON string.

    This node handles various types of input:
    - Python dictionaries or lists: These will be serialized directly into a JSON string.
    - Valid JSON strings: These will be parsed, validated, and then re-serialized,
      allowing for consistent formatting (e.g., pretty-printing with indentation).
    - Other JSON-serializable Python types (e.g., int, float, bool, None, simple strings):
      These will be directly serialized to their respective JSON representations.

    If the input data cannot be successfully converted into a valid JSON string,
    the node will log a detailed error and return None, indicating a processing failure.
    """

    def __init__(self, indent: Optional[int] = None):
        """
        Initializes the JSONFormatterNode.

        Args:
            indent: An optional integer that specifies the indentation level for
                    pretty-printing the JSON output. If None, the JSON will be
                    compact (no extra whitespace).
        """
        self._indent = indent
        logger.debug(f"JSONFormatterNode initialized with indent={indent}")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "JSONFormatter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, attempting to convert it into a formatted JSON string.

        The `context` parameter is available for potential future use by the framework
        but is not directly utilized by this node for its core formatting logic.

        Args:
            data: The input data to be formatted. This can be a dict, list, a string
                  containing JSON, or other JSON-serializable primitive types.
            context: A dictionary containing contextual information relevant to the
                     current orchestration run.

        Returns:
            A formatted JSON string if the operation is successful. Returns None if
            the input data cannot be converted into a valid JSON string, after logging
            the encountered error.
        """
        logger.debug(f"JSONFormatterNode received data of type: {type(data)}")

        # Start with the assumption that the input data is directly serializable.
        # This will be overridden if the data is a string that needs parsing.
        data_to_serialize = data

        # If the input is a string, attempt to parse it first.
        # This allows us to re-format existing JSON strings (e.g., apply indentation).
        if isinstance(data, str):
            if not data.strip(): # Handle empty or whitespace-only strings gracefully
                logger.warning("JSONFormatterNode received an empty or whitespace-only string, cannot parse as JSON.")
                return None
            try:
                data_to_serialize = json.loads(data)
                logger.debug("Successfully parsed input string as a JSON object for re-serialization.")
            except json.JSONDecodeError as e:
                # If the string is not valid JSON, we log it, but still try to serialize
                # the *original string itself* (e.g., "plain text" -> "\"plain text\"").
                logger.warning(
                    f"JSONFormatterNode input string could not be parsed as valid JSON: {e}. "
                    "Attempting to serialize the original string directly as a JSON string literal."
                )
                data_to_serialize = data # Revert to original string for direct serialization
            except Exception as e:
                logger.error(f"An unexpected error occurred during JSON string parsing: {e}", exc_info=True)
                return None # Catch any other unexpected parsing errors

        # Attempt to serialize the (potentially parsed) Python object to a JSON string.
        try:
            formatted_json = json.dumps(data_to_serialize, indent=self._indent)
            logger.info("Successfully formatted data as JSON.")
            return formatted_json
        except TypeError as e:
            # This error occurs if the Python object is not JSON-serializable
            # (e.g., a set, a complex object without a custom serializer).
            logger.error(
                f"JSONFormatterNode failed to serialize data to JSON due to a TypeError: {e}. "
                f"Data type: {type(data_to_serialize)}. Data sample: {str(data_to_serialize)[:200]}..."
            )
            return None
        except Exception as e:
            # Catch any other unexpected errors during the serialization process.
            logger.error(f"An unexpected error occurred during JSON serialization: {e}", exc_info=True)
            return None