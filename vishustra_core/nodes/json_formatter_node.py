import json
import logging
from typing import Any, Dict, Optional

# Assuming this path exists in the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JsonFormatterNode(BaseNode):
    """
    A processing node that formats input data into a JSON string.

    This node attempts to serialize the input data into a JSON string.
    If the input is already a string that represents valid JSON, it will
    be re-parsed and then re-serialized to apply formatting (e.g., indentation).
    If the input string is not valid JSON, it will be treated as a literal
    string to be JSON-encoded (e.g., "hello" becomes '"hello"').
    Non-serializable objects will result in an error log and a JSON string
    indicating the failure.
    """

    def __init__(self, indent: Optional[int] = None):
        """
        Initializes the JsonFormatterNode.

        Args:
            indent (Optional[int]): If provided, the JSON output will be
                                    pretty-printed with the specified indent level.
                                    If None, the output will be compact.
        """
        self._indent = indent

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "JsonFormatterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data, formatting it into a JSON string.

        Args:
            data (Any): The data to be formatted into JSON.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the current processing flow. Not directly
                                       used for formatting in this node, but passed
                                       as per BaseNode contract.

        Returns:
            str: A JSON string representation of the input data.
                 If data is not JSON serializable, returns a JSON string
                 indicating the error.
        """
        try:
            if isinstance(data, str):
                try:
                    # If the string is already valid JSON, parse it to re-dump
                    # with proper formatting.
                    parsed_data = json.loads(data)
                    return json.dumps(parsed_data, indent=self._indent)
                except json.JSONDecodeError:
                    # If the string is not valid JSON, treat it as a literal
                    # string value to be JSON encoded (e.g., "text" -> '"text"')
                    logger.debug(
                        f"[{self.node_name}] Input string is not valid JSON, "
                        f"treating as literal string: '{data[:50]}...'"
                    )
                    return json.dumps(data, indent=self._indent)
            else:
                # For dicts, lists, numbers, etc., directly dump them
                return json.dumps(data, indent=self._indent)
        except TypeError as e:
            error_message = (
                f"[{self.node_name}] Data of type '{type(data).__name__}' "
                f"is not JSON serializable: {e}"
            )
            logger.error(error_message)
            # Return a structured JSON error response
            return json.dumps(
                {
                    "error": "NonSerializableData",
                    "message": "Input data could not be serialized to JSON.",
                    "detail": str(e),
                    "input_type": str(type(data).__name__),
                },
                indent=self._indent,
            )
        except Exception as e:
            # Catch any other unexpected errors during processing
            error_message = (
                f"[{self.node_name}] An unexpected error occurred "
                f"during JSON formatting: {e}"
            )
            logger.exception(error_message)
            return json.dumps(
                {
                    "error": "ProcessingError",
                    "message": "An unexpected error occurred during JSON formatting.",
                    "detail": str(e),
                    "input_type": str(type(data).__name__),
                },
                indent=self._indent,
            )

# Example usage (for testing purposes, not part of Vishustra execution)
if __name__ == "__main__":
    # Configure basic logging for console output
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.DEBUG) # Set this specific logger to DEBUG for more detail

    # Create an instance of the formatter with pretty printing
    pretty_formatter = JsonFormatterNode(indent=2)
    compact_formatter = JsonFormatterNode()

    test_context = {}

    # Test Case 1: Python dictionary
    data_dict = {"name": "Alice", "age": 30, "is_student": False, "courses": ["Math", "Science"]}
    print("\n--- Python Dict (Pretty) ---")
    print(pretty_formatter.process(data_dict, test_context))
    print("\n--- Python Dict (Compact) ---")
    print(compact_formatter.process(data_dict, test_context))

    # Test Case 2: String that is valid JSON
    json_string = '{"city": "New York", "population": 8000000}'
    print("\n--- Valid JSON String (Pretty) ---")
    print(pretty_formatter.process(json_string, test_context))

    # Test Case 3: String that is NOT valid JSON
    plain_string = "Hello, Vishustra users!"
    print("\n--- Plain String (Pretty) ---")
    print(pretty_formatter.process(plain_string, test_context))

    # Test Case 4: List of items
    data_list = [1, 2, {"a": 1, "b": 2}, "three"]
    print("\n--- Python List (Pretty) ---")
    print(pretty_formatter.process(data_list, test_context))

    # Test Case 5: Non-serializable object
    class NonSerializable:
        def __init__(self):
            self.value = "I cannot be JSON-ified directly"

    non_serializable_obj = NonSerializable()
    print("\n--- Non-Serializable Object (Pretty) ---")
    print(pretty_formatter.process(non_serializable_obj, test_context))

    # Test Case 6: Integer
    data_int = 12345
    print("\n--- Integer (Compact) ---")
    print(compact_formatter.process(data_int, test_context))