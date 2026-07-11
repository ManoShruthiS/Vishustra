import logging
from typing import Any, Dict

# Assuming BaseNode is available at this path in the Vishustra core
# For local testing, one might create a dummy 'vishustra_core/nodes/base_node.py'
# with the BaseNode class definition.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra processing node responsible for translating text content
    from a source language to a target language.

    This node simulates translation functionality, expecting the input `data`
    to be a string and `context` to contain 'target_language' and optionally
    'source_language'. In a production environment, this would integrate
    with an external translation service (e.g., Google Translate API, DeepL).
    """

    def __init__(self):
        """
        Initializes the LanguageTranslatorNode.
        A real implementation might load translation models or API keys here.
        """
        # A very basic, illustrative translation map for demonstration purposes.
        # In a real scenario, this would be an external service call.
        self._translation_map = {
            "en": {
                "hello": {"es": "hola", "fr": "bonjour", "de": "hallo"},
                "world": {"es": "mundo", "fr": "monde", "de": "welt"},
                "goodbye": {"es": "adiós", "fr": "au revoir", "de": "auf wiedersehen"},
                "thank you": {"es": "gracias", "fr": "merci", "de": "danke schön"},
                "vishustra": {"es": "vishustra", "fr": "vishustra", "de": "vishustra"},
                "this is": {"es": "esto es", "fr": "c'est", "de": "das ist"},
            }
        }
        logger.debug("LanguageTranslatorNode initialized with dummy translation map.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "LanguageTranslatorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Translates the input text `data` based on the specified languages in `context`.

        Args:
            data: The input text to be translated (expected to be a string).
            context: A dictionary containing processing parameters.
                     Expected keys:
                     - 'target_language' (str): The ISO 639-1 code of the target language
                                                (e.g., 'es', 'fr', 'de').
                     - 'source_language' (str, optional): The ISO 639-1 code of the source
                                                          language (defaults to 'en' for this
                                                          simulation if not provided).

        Returns:
            str: The translated text.

        Raises:
            ValueError: If `data` is not a string, `target_language` is missing,
                        or an unsupported language is specified.
            RuntimeError: If an unexpected issue occurs during the simulated translation.
        """
        logger.info(f"[{self.node_name}] Starting process for data type: {type(data)}")
        logger.debug(f"[{self.node_name}] Context: {context}")

        if not isinstance(data, str):
            logger.error(f"[{self.node_name}] Invalid input data type. Expected str, got {type(data)}.")
            raise ValueError(
                f"LanguageTranslatorNode expects string input data, but received {type(data).__name__}."
            )

        target_language = context.get("target_language")
        source_language = context.get("source_language", "en") # Default to English for simulation

        if not target_language:
            logger.error(f"[{self.node_name}] 'target_language' not found in context.")
            raise ValueError(
                "LanguageTranslatorNode requires 'target_language' in the processing context."
            )

        if source_language not in self._translation_map:
            logger.error(f"[{self.node_name}] Unsupported source language: '{source_language}'.")
            raise ValueError(
                f"Unsupported source language '{source_language}' for translation simulation."
            )

        if target_language not in self._translation_map[source_language]:
            logger.error(f"[{self.node_name}] Unsupported target language: '{target_language}'.")
            raise ValueError(
                f"Unsupported target language '{target_language}' for translation simulation from '{source_language}'."
            )

        translated_parts = []
        # A simple word-by-word/phrase-by-phrase simulation
        # In a real system, this would be sentence-level or even document-level translation.
        # This split is very basic and won't handle punctuation perfectly.
        words = data.split()
        
        for word in words:
            # Normalize word for lookup (simple lowercasing)
            lower_word = word.lower()
            translation = self._translation_map[source_language].get(lower_word, {}).get(target_language)
            if translation:
                # Retain original casing if possible, or just use the translated word
                if word[0].isupper() and len(word) > 0:
                    translated_parts.append(translation.capitalize())
                else:
                    translated_parts.append(translation)
                logger.debug(f"[{self.node_name}] Translated '{word}' to '{translation}' in '{target_language}'.")
            else:
                # If no direct translation found, keep the original word
                translated_parts.append(word)
                logger.debug(f"[{self.node_name}] No translation found for '{word}' in '{target_language}', keeping original.")
        
        translated_text = " ".join(translated_parts)

        logger.info(
            f"[{self.node_name}] Translated text from '{source_language}' to '{target_language}'. "
            f"Original: '{data[:50]}...' -> Translated: '{translated_text[:50]}...'"
        )
        return translated_text

# Example of how to use this node (for testing purposes, not part of the file)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Dummy BaseNode for local testing
    class BaseNode(BaseNode):
        def process(self, data: Any, context: Dict[str, Any]) -> Any:
            pass
        @property
        def node_name(self) -> str:
            pass

    translator = LanguageTranslatorNode()

    # Test cases
    try:
        # Valid translation
        text_en = "Hello world! This is Vishustra."
        context_es = {"target_language": "es", "source_language": "en"}
        translated_es = translator.process(text_en, context_es)
        print(f"Original (EN): {text_en}")
        print(f"Translated (ES): {translated_es}\n")
        assert translated_es == "Hola mundo! Esto es Vishustra."

        text_fr = "Thank you, Vishustra."
        context_fr = {"target_language": "fr"} # source_language defaults to 'en'
        translated_fr = translator.process(text_fr, context_fr)
        print(f"Original (EN): {text_fr}")
        print(f"Translated (FR): {translated_fr}\n")
        assert translated_fr == "Merci, Vishustra."

        # Invalid input type
        try:
            translator.process(123, context_es)
        except ValueError as e:
            print(f"Caught expected error: {e}\n")
        
        # Missing target language
        try:
            translator.process(text_en, {})
        except ValueError as e:
            print(f"Caught expected error: {e}\n")

        # Unsupported target language (in our dummy map)
        try:
            translator.process(text_en, {"target_language": "ja"})
        except ValueError as e:
            print(f"Caught expected error: {e}\n")

    except Exception as e:
        logger.critical(f"An unexpected error occurred during testing: {e}")
