import logging
from typing import Any, Dict, List, Union

# Simulate the import path for BaseNode within Vishustra
# In a real project, this would be:
# from vishustra_core.nodes.base_node import BaseNode
# For standalone execution/testing, we define a mock BaseNode if not available:
try:
    from vishustra_core.nodes.base_node import BaseNode
except ImportError:
    from abc import ABC, abstractmethod

    class BaseNode(ABC):
        """
        Mock Base class for Vishustra processing nodes.
        Used when vishustra_core is not directly available in the environment.
        """
        @abstractmethod
        def process(self, data: Any, context: Dict[str, Any]) -> Any:
            """Processes the input data and returns the result."""
            pass

        @property
        @abstractmethod
        def node_name(self) -> str:
            """Returns the name of the node."""
            pass

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra processing node that filters out profanity from text data.

    This node is designed to sanitize strings or lists of strings by replacing
    identified profanities with asterisks. It operates case-insensitively.
    """

    def __init__(self):
        """
        Initializes the ProfanityFilterNode with a predefined list of profanities.
        In a production system, this list would likely be externalized
        (e.g., from configuration, a database, or a specialized service).
        """
        self._profanity_list: List[str] = [
            "fuck", "shit", "asshole", "bitch", "cunt", "damn", "pussy", "dick",
            "bastard", "motherfucker"
        ]
        logger.info(f"ProfanityFilterNode initialized with {len(self._profanity_list)} profanities.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ProfanityFilterNode"

    def _replace_profanity(self, text: str) -> str:
        """
        Helper method to replace profanity within a single string.
        Matches are case-insensitive and replaced with '*' characters of the same length.
        """
        processed_text = text
        text_lower = text.lower()

        for profanity in self._profanity_list:
            # Simple replacement; more advanced filters would use regex for word boundaries
            # and avoid replacing parts of words (e.g., "classic" -> "cl*****c")
            if profanity in text_lower:
                replacement = '*' * len(profanity)
                # Find all occurrences and replace them. A simple `replace` might be
                # too naive if the original casing needs to be preserved for non-profane parts.
                # For this implementation, we replace the lowercase match in the original text.
                # A more robust approach would use regex with re.sub(pattern, repl, string, flags=re.IGNORECASE)
                # For simplicity here, we'll replace the exact matched string with replacement.
                # This could lead to imperfect casing if "FUCK" becomes "****", but "fuck" also becomes "****".
                # A better approach for preserving casing while replacing:
                start_index = 0
                while True:
                    match_index = text_lower.find(profanity, start_index)
                    if match_index == -1:
                        break
                    processed_text = (
                        processed_text[:match_index] +
                        replacement +
                        processed_text[match_index + len(profanity):]
                    )
                    start_index = match_index + len(profanity)
        return processed_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to filter out profanity.

        Args:
            data: The input data, expected to be a string or a list of strings.
            context: A dictionary containing contextual information for processing.
                     (Not directly used by this node but passed through pipeline).

        Returns:
            The processed data with profanities filtered, or the original data
            if it's not a supported type.

        Raises:
            TypeError: If the input data is not a string or a list of strings.
        """
        logger.debug(f"[{self.node_name}] Starting process for data type: {type(data)}")

        if isinstance(data, str):
            filtered_data = self._replace_profanity(data)
            logger.debug(f"[{self.node_name}] Processed single string.")
            return filtered_data
        elif isinstance(data, list) and all(isinstance(item, str) for item in data):
            filtered_data = [self._replace_profanity(item) for item in data]
            logger.debug(f"[{self.node_name}] Processed list of strings.")
            return filtered_data
        else:
            error_msg = (
                f"[{self.node_name}] Invalid input type for profanity filtering. "
                f"Expected str or List[str], but received {type(data)}."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

# Example usage (for testing purposes, not part of the required output):
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    profanity_node = ProfanityFilterNode()

    test_context = {"user_id": "test_user"}

    # Test cases
    test_string_1 = "This is a damn good example, you motherfucker!"
    test_string_2 = "No bad words here. Just a nice sentence."
    test_string_3 = "WHAT THE FUCK IS THIS SHIT?"
    test_string_4 = "The quick brown fox jumps over the lazy dog."

    test_list_1 = [
        "That's a load of shit.",
        "I love this amazing project!",
        "Don't be an asshole, follow the rules."
    ]
    test_list_2 = ["Pure text, no issues.", "Another clean entry."]

    test_invalid_int = 123
    test_invalid_dict = {"text": "hello"}

    print(f"\nNode Name: {profanity_node.node_name}")

    print("\n--- Testing single strings ---")
    print(f"Original: '{test_string_1}'")
    print(f"Filtered: '{profanity_node.process(test_string_1, test_context)}'")

    print(f"\nOriginal: '{test_string_2}'")
    print(f"Filtered: '{profanity_node.process(test_string_2, test_context)}'")

    print(f"\nOriginal: '{test_string_3}'")
    print(f"Filtered: '{profanity_node.process(test_string_3, test_context)}'")

    print(f"\nOriginal: '{test_string_4}'")
    print(f"Filtered: '{profanity_node.process(test_string_4, test_context)}'")

    print("\n--- Testing lists of strings ---")
    print(f"Original: {test_list_1}")
    print(f"Filtered: {profanity_node.process(test_list_1, test_context)}")

    print(f"\nOriginal: {test_list_2}")
    print(f"Filtered: {profanity_node.process(test_list_2, test_context)}")

    print("\n--- Testing invalid inputs ---")
    try:
        print(f"Original (int): {test_invalid_int}")
        profanity_node.process(test_invalid_int, test_context)
    except TypeError as e:
        print(f"Caught expected error: {e}")

    try:
        print(f"Original (dict): {test_invalid_dict}")
        profanity_node.process(test_invalid_dict, test_context)
    except TypeError as e:
        print(f"Caught expected error: {e}")

    try:
        print(f"Original (list with non-str): {[1, 'test']}")
        profanity_node.process([1, 'test'], test_context)
    except TypeError as e:
        print(f"Caught expected error: {e}")
