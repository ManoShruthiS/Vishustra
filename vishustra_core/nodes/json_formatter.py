import json
import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JSONFormatter(BaseNode):
    """
    A processing node that serializes input data into a pretty-printed JSON string.

    This node intelligently handles various input types:
    - If the input `data` is already a Python dictionary or list, it is directly
      serialized to a JSON string.
    - If the input `data` is a string, the node first attempts to parse it as JSON.
      If successful, the parsed object is then pretty-printed. If the string is
      not valid JSON, the original string itself is serialized (e.g., '"my string"').
    - Non-serializable objects (e.g., custom class instances without a `to_json` method
      or similar serialization hook) will cause a `TypeError`, which is caught.
      In such cases, an error is logged, and the original `data` is returned
      to avoid pipeline interruption.

    Configuration for indentation can be provided via the `context` dictionary.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "JSONFormatter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Formats the input data as a pretty-printed JSON string.

        Args:
            data: The input data to be formatted. This can be any Python object
                  that is JSON-serializable, or a string that may or may not be
                  valid JSON.
            context: A dictionary containing contextual information and node-specific
                     configuration.
                     Recognized keys:
                     - 'indent': (int, optional) The number of spaces to use for
                                 indentation in the JSON output. Defaults to 2.

        Returns:
            A pretty-printed JSON string. If the input data is not JSON-serializable,
            the original `data` is returned, and an error is logged.
        """
        # Retrieve indentation level from context, default to 2 spaces
        indent = context.get('indent', 2)
        
        # Validate 'indent' value
        if not isinstance(indent, int) or indent < 0:
            logger.warning(
                f"[{self.node_name}] Invalid 'indent' value '{indent}' ({type(indent).__name__}) "
                "provided in context. Falling back to default indent of 2."
            )
            indent = 2

        data_to_serialize = data

        # If the input is a string, try to load it as JSON first
        if isinstance(data, str):
            try:
                data_to_serialize = json.loads(data)
                logger.debug(f"[{self.node_name}] Successfully parsed input string as JSON.")
            except json.JSONDecodeError:
                logger.debug(
                    f"[{self.node_name}] Input string is not valid JSON. "
                    "Attempting to serialize the original string directly."
                )
                # If not valid JSON, proceed to serialize the string itself
                pass
            except Exception as e:
                logger.error(
                    f"[{self.node_name}] Unexpected error while attempting to parse input string as JSON: {e}"
                )
                # Fallback to original string if an unexpected error occurs during parsing
                pass

        try:
            # Serialize the data (either the parsed object or the original data)
            # ensure_ascii=False ensures non-ASCII characters are output directly
            # instead of being escaped (\uXXXX).
            return json.dumps(data_to_serialize, indent=indent, ensure_ascii=False)
        except TypeError as e:
            logger.error(
                f"[{self.node_name}] Data of type '{type(data_to_serialize).__name__}' "
                f"is not JSON serializable. Returning original data. Error: {e}"
            )
            return data
        except Exception as e:
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during JSON serialization: {e}"
            )
            return data

