import json
import logging
from typing import Any, Dict

# BaseNode is provided as context in the problem description, 
# but the instruction is to import it from the specified path.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JSONFormatterNode(BaseNode):
    """
    A Vishustra processing node that formats input data into a JSON string.
    
    This node first attempts to convert the input `data` into a Python object.
    If `data` is a string, it tries to parse it as JSON; otherwise, it treats 
    the input directly as a Python object. After obtaining a Python object,
    it serializes it back into a formatted JSON string using `json.dumps`.
    
    Configuration options can be provided via the `context` dictionary:
    - 'indent' (int or None): The number of spaces to use for indentation. 
      If None, produces a compact JSON string. Default is 2 spaces for pretty-printing.
    - 'sort_keys' (bool): If true, output dictionary keys are sorted alphabetically. 
      Default is False.
    """
    
    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "JSONFormatter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, attempting to format it as a JSON string.
        
        Args:
            data (Any): The input data to be formatted. This can be a Python 
                        object (dict, list, str, int, float, bool, None) or 
                        a JSON string.
            context (Dict[str, Any]): A dictionary containing runtime context 
                                     and potential configuration parameters.
                                     Expected keys: 'indent' (int|None), 
                                     'sort_keys' (bool).
                                     
        Returns:
            str: A formatted JSON string.
            
        Raises:
            ValueError: If the input data cannot be parsed into a Python object 
                        or serialized into a JSON string.
        """
        node_id = context.get('node_id', self.node_name)
        logger.debug(f"[{node_id}] Starting JSONFormatterNode process with data type: {type(data)}")

        # Determine formatting options from context with sensible defaults
        indent_level = context.get('indent', 2) # Default to 2 spaces for pretty-printing
        sort_output_keys = context.get('sort_keys', False) # Default to not sorting keys

        python_object_to_serialize: Any = None

        try:
            if isinstance(data, str):
                try:
                    # Attempt to parse string data as JSON
                    python_object_to_serialize = json.loads(data)
                    logger.debug(f"[{node_id}] Successfully parsed input string as a JSON object.")
                except json.JSONDecodeError:
                    # If string is not valid JSON, treat it as a literal string to be serialized
                    python_object_to_serialize = data
                    logger.debug(f"[{node_id}] Input string is not valid JSON; treating as a literal string to be formatted.")
            else:
                # For non-string data (dict, list, int, bool, None, custom objects),
                # use it directly for serialization. `json.dumps` will handle basic types.
                # For custom objects, `json.dumps` will raise TypeError if not serializable.
                python_object_to_serialize = data
                logger.debug(f"[{node_id}] Input data is not a string, proceeding to serialize directly.")

            # Serialize the Python object (or literal string) into a JSON string
            formatted_json_string = json.dumps(
                python_object_to_serialize,
                indent=indent_level,
                sort_keys=sort_output_keys,
                ensure_ascii=False # Often preferred for modern applications to output non-ASCII characters directly
            )
            
            logger.info(f"[{node_id}] Successfully formatted data into JSON string.")
            return formatted_json_string

        except (TypeError, OverflowError) as e:
            # TypeError for non-serializable objects (e.g., custom class instances without a __dict__ or default handler)
            # OverflowError for extremely large int/float not representable in JSON
            logger.error(f"[{node_id}] Input data of type {type(data)} is not JSON serializable: {e}")
            raise ValueError(f"[{node_id}] Input data is not JSON serializable: {e}") from e
        except Exception as e:
            # Catch any other unexpected errors during the process
            logger.error(f"[{node_id}] An unexpected error occurred during JSON formatting: {e}")
            raise ValueError(f"[{node_id}] An unexpected error occurred: {e}") from e