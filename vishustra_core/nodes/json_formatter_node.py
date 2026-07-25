import json
import logging
from typing import Any, Dict

# Assuming BaseNode is in this path as per project context and instructions
from vishustra_core.nodes.base_node import BaseNode 

logger = logging.getLogger(__name__)

class JSONFormatterNode(BaseNode):
    """
    A Vishustra node that formats input data into a JSON string.
    
    This node offers versatile handling for various input types:
    - Python objects (dictionaries, lists, custom objects if serializable) are converted into a JSON string.
    - Existing JSON strings are parsed and then re-serialized with the specified formatting options.
    - Plain strings (that are not valid JSON) are treated as a literal string and serialized accordingly.
    
    Context options:
    The 'context' dictionary can include a 'json_formatter_options' key.
    The value associated with this key should be a dictionary, where keys and values
    correspond to arguments accepted by Python's `json.dumps()` function.
    Commonly used options include 'indent' (int for pretty-printing) and 'sort_keys' (bool).
    
    Example context usage for pretty-printing and sorting keys:
    ```python
    context = {
        'json_formatter_options': {
            'indent': 4,
            'sort_keys': True
        }
    }
    ```
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "JSONFormatterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Formats the input data into a JSON string using configurable options.

        Args:
            data: The input data to be formatted. This can be any Python object
                  that `json.dumps` can serialize, an existing JSON string,
                  or a regular string.
            context: A dictionary for contextual information. It can contain
                     a 'json_formatter_options' dictionary to pass arguments
                     like 'indent' and 'sort_keys' to `json.dumps()`.

        Returns:
            A formatted JSON string upon successful serialization, otherwise None.
        """
        logger.debug(f"[{self.node_name}] Starting process for data of type: {type(data)}.")

        # Extract JSON serialization options from the context
        json_dumps_options = context.get('json_formatter_options', {})
        
        # Prepare data for serialization.
        # If the input is a string, attempt to parse it first.
        # This allows re-formatting existing JSON strings.
        serializable_data = data
        if isinstance(data, str):
            try:
                parsed_data = json.loads(data)
                serializable_data = parsed_data
                logger.debug(f"[{self.node_name}] Input string successfully parsed as a JSON object.")
            except json.JSONDecodeError:
                # If the string is not valid JSON, we treat it as a literal string
                # which `json.dumps` will correctly escape and quote.
                logger.debug(f"[{self.node_name}] Input string is not valid JSON; treating as a literal string for serialization.")
            except Exception as e:
                # Catch any other unexpected errors during string parsing
                logger.warning(
                    f"[{self.node_name}] Unexpected error while attempting to parse input string as JSON: {type(e).__name__}: {e}. "
                    "Proceeding with original string data."
                )
                serializable_data = data # Fallback to original data

        try:
            # Perform the JSON serialization with the collected options
            formatted_json_string = json.dumps(serializable_data, **json_dumps_options)
            logger.info(f"[{self.node_name}] Data successfully formatted to JSON.")
            return formatted_json_string
        except TypeError as e:
            logger.error(
                f"[{self.node_name}] Failed to serialize data. Input of type "
                f"'{type(serializable_data).__name__}' is not JSON serializable: {e}"
            )
            return None
        except json.JSONEncodeError as e:
            logger.error(
                f"[{self.node_name}] Failed to encode data to JSON due to an encoding error: {e}"
            )
            return None
        except Exception as e:
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during JSON formatting: {type(e).__name__}: {e}"
            )
            return None