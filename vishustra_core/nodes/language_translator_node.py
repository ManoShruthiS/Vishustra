import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra node responsible for simulating language translation of text data.

    This node expects a string as input data and a 'target_language' key in the
    context dictionary to specify the language to translate to.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "LanguageTranslatorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Simulates the translation of input text data to a specified target language.

        Args:
            data (Any): The input data to be translated. Expected to be a string.
            context (Dict[str, Any]): A dictionary containing contextual information,
                                       expected to include 'target_language'.

        Returns:
            Any: The simulated translated text (a string).

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If 'target_language' is missing or not a string in the context.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', got '{type(data).__name__}'."
            )
            raise TypeError(
                f"[{self.node_name}] Input data must be a string for translation. "
                f"Received type: {type(data).__name__}"
            )

        target_language = context.get("target_language")
        if not isinstance(target_language, str) or not target_language:
            logger.error(
                f"[{self.node_name}] 'target_language' is missing or invalid in context. "
                "It must be a non-empty string."
            )
            raise ValueError(
                f"[{self.node_name}] 'target_language' must be provided as a non-empty string in the context. "
                f"Received: {target_language}"
            )

        logger.info(
            f"[{self.node_name}] Attempting to simulate translation of data to '{target_language}'."
        )

        # Simulate translation by appending a tag. In a real scenario, this would
        # involve an external translation API call.
        translated_data = f"{data} [Translated to {target_language}]"

        logger.debug(
            f"[{self.node_name}] Successfully simulated translation to '{target_language}'. "
            f"Original length: {len(data)}, Translated length: {len(translated_data)}"
        )
        return translated_data

# Example usage (for internal testing, not part of the module's core functionality)
if __name__ == '__main__':
    # Configure basic logging for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    translator_node = LanguageTranslatorNode()

    # Test Case 1: Successful translation
    text_to_translate = "Hello, world!"
    translation_context = {"target_language": "Spanish"}
    try:
        translated_result = translator_node.process(text_to_translate, translation_context)
        print(f"Original: '{text_to_translate}'")
        print(f"Translated: '{translated_result}'")
        assert translated_result == "Hello, world! [Translated to Spanish]"
    except Exception as e:
        print(f"Error during successful translation test: {e}")

    print("\n--- Testing error cases ---")

    # Test Case 2: Missing target_language
    text_to_translate_error = "This should fail."
    missing_lang_context = {"source_language": "English"}
    try:
        translator_node.process(text_to_translate_error, missing_lang_context)
    except ValueError as e:
        print(f"Caught expected error (missing target_language): {e}")
    except Exception as e:
        print(f"Caught unexpected error type: {type(e).__name__} - {e}")

    # Test Case 3: Invalid data type
    invalid_data = 12345
    valid_lang_context = {"target_language": "French"}
    try:
        translator_node.process(invalid_data, valid_lang_context)
    except TypeError as e:
        print(f"Caught expected error (invalid data type): {e}")
    except Exception as e:
        print(f"Caught unexpected error type: {type(e).__name__} - {e}")

    # Test Case 4: Empty target_language
    empty_lang_context = {"target_language": ""}
    try:
        translator_node.process(text_to_translate, empty_lang_context)
    except ValueError as e:
        print(f"Caught expected error (empty target_language): {e}")
    except Exception as e:
        print(f"Caught unexpected error type: {type(e).__name__} - {e}")