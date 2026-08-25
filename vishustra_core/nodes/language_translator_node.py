import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslator(BaseNode):
    """
    A Vishustra node that simulates the translation of text from a source
    language to a specified target language.

    This node expects the input 'data' to be a string and the 'context'
    dictionary to contain a 'target_language' key with a string value.
    Optionally, 'source_language' can also be provided in the context.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "LanguageTranslator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by simulating language translation.

        Args:
            data: The input text to be translated (expected to be a string).
            context: A dictionary containing operational parameters,
                     expected to have 'target_language' (str) and
                     optionally 'source_language' (str).

        Returns:
            The translated string.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If 'target_language' is missing or invalid in the context.
        """
        logger.debug(f"[{self.node_name}] Starting process for data: '{data[:50]}...' with context: {context}")

        # Validate input data type
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', but received '{type(data).__name__}'."
            )
            raise TypeError(f"'{self.node_name}' node expects 'data' to be a string.")

        # Validate and retrieve target language from context
        target_language = context.get("target_language")
        if not target_language or not isinstance(target_language, str):
            logger.error(
                f"[{self.node_name}] Missing or invalid 'target_language' in context. "
                "Context must contain a valid string for 'target_language'."
            )
            raise ValueError(
                f"'{self.node_name}' node requires 'target_language' (str) in context."
            )

        source_language = context.get("source_language", "auto")
        logger.debug(
            f"[{self.node_name}] Attempting to translate from '{source_language}' to '{target_language}'."
        )

        translated_text: str

        if not data.strip():
            # Handle empty or whitespace-only strings
            translated_text = data
            logger.info(f"[{self.node_name}] Received empty or whitespace-only data, returning as-is.")
        else:
            # Simulate translation
            # In a real scenario, this would involve calling a translation API
            # For this simulation, we'll append a tag or use simple mappings
            _target_lang_lower = target_language.lower()
            _data_lower = data.lower()

            if _target_lang_lower == 'es':
                if _data_lower == 'hello':
                    translated_text = 'Hola'
                elif _data_lower == 'goodbye':
                    translated_text = 'Adiós'
                else:
                    translated_text = f"{data} (traducido al español)"
            elif _target_lang_lower == 'fr':
                if _data_lower == 'hello':
                    translated_text = 'Bonjour'
                elif _data_lower == 'goodbye':
                    translated_text = 'Au revoir'
                else:
                    translated_text = f"{data} (traduit en français)"
            else:
                # Generic fallback for other languages
                translated_text = f"{data} (translated to {target_language})"

        logger.info(
            f"[{self.node_name}] Successfully translated text. "
            f"Original (first 30 chars): '{data[:30]}' -> Translated (first 30 chars): '{translated_text[:30]}'"
        )
        return translated_text

# Example of how to configure logging (typically done once at application startup)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    translator = LanguageTranslator()

    # Test cases
    print(f"\n--- Testing '{translator.node_name}' node ---")

    # Valid translation
    try:
        translated_es = translator.process("Hello Vishustra team!", {"target_language": "es"})
        print(f"Original: 'Hello Vishustra team!' -> Translated (ES): '{translated_es}'")
    except Exception as e:
        print(f"Error during translation: {e}")

    try:
        translated_fr = translator.process("Goodbye for now.", {"target_language": "fr"})
        print(f"Original: 'Goodbye for now.' -> Translated (FR): '{translated_fr}'")
    except Exception as e:
        print(f"Error during translation: {e}")

    # Specific word translation
    try:
        translated_hello_es = translator.process("Hello", {"target_language": "es"})
        print(f"Original: 'Hello' -> Translated (ES): '{translated_hello_es}'")
    except Exception as e:
        print(f"Error during translation: {e}")

    # Unsupported language simulation
    try:
        translated_ja = translator.process("Welcome to Vishustra", {"target_language": "ja"})
        print(f"Original: 'Welcome to Vishustra' -> Translated (JA): '{translated_ja}'")
    except Exception as e:
        print(f"Error during translation: {e}")

    # Empty string data
    try:
        translated_empty = translator.process("", {"target_language": "en"})
        print(f"Original: '' -> Translated (EN): '{translated_empty}'")
    except Exception as e:
        print(f"Error during translation: {e}")

    # Error case: Missing target_language
    print("\n--- Testing error cases ---")
    try:
        translator.process("Some text", {"source_language": "en"})
    except ValueError as e:
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"Caught unexpected error: {type(e).__name__} - {e}")

    # Error case: Invalid data type
    try:
        translator.process(123, {"target_language": "en"})
    except TypeError as e:
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"Caught unexpected error: {type(e).__name__} - {e}")

    # Error case: Invalid target_language type
    try:
        translator.process("Hello", {"target_language": 123})
    except ValueError as e:
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"Caught unexpected error: {type(e).__name__} - {e}")

