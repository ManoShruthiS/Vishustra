import logging
from typing import Any, Dict

# Assuming BaseNode is located in vishustra_core.nodes.base_node relative to the project root
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

# A simple, static translation map for demonstration purposes.
# In a real-world scenario, this would integrate with an actual
# translation service (e.g., Google Translate API, DeepL, etc.).
_SIMULATED_TRANSLATION_MAP = {
    "hello": {"es": "hola", "fr": "bonjour", "de": "hallo"},
    "world": {"es": "mundo", "fr": "monde", "de": "welt"},
    "vishustra": {"es": "vishustra", "fr": "vishustra", "de": "vishustra"},
    "node": {"es": "nodo", "fr": "nœud", "de": "knoten"},
    "framework": {"es": "marco", "fr": "cadre", "de": "rahmenwerk"},
    "process": {"es": "procesar", "fr": "traiter", "de": "verarbeiten"},
    "data": {"es": "datos", "fr": "données", "de": "daten"},
    "great": {"es": "excelente", "fr": "excellent", "de": "großartig"},
    "example": {"es": "ejemplo", "fr": "exemple", "de": "beispiel"},
}

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra processing node that simulates language translation of text data.

    This node expects the input 'data' to be a string and the 'context'
    dictionary to contain a 'target_language' key with the desired
    language code (e.g., 'es', 'fr', 'de').

    For demonstration, it uses a simple hardcoded translation map.
    In a production environment, this would integrate with a robust
    third-party translation service.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "LanguageTranslator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Translates the input text data to the specified target language.

        Args:
            data: The input data, expected to be a string of text to translate.
            context: A dictionary containing operational context, expected to have
                     a 'target_language' key (e.g., 'es', 'fr', 'de').

        Returns:
            The translated text as a string, or the original data if translation
            is not possible (e.g., unsupported language, non-string input).

        Raises:
            ValueError: If `data` is not a string.
            KeyError: If 'target_language' is missing from `context`.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected string, got {type(data)}. "
                "Returning original data."
            )
            raise ValueError(f"LanguageTranslatorNode expects string data, got {type(data)}")

        try:
            target_language = context["target_language"]
        except KeyError:
            logger.error(
                f"[{self.node_name}] Missing 'target_language' in context. "
                "Translation cannot proceed. Returning original data."
            )
            raise KeyError("Context must contain 'target_language' for LanguageTranslatorNode.")

        if not isinstance(target_language, str) or not target_language:
            logger.warning(
                f"[{self.node_name}] 'target_language' in context is invalid or empty: '{target_language}'. "
                "Returning original data."
            )
            return data

        logger.info(f"[{self.node_name}] Attempting to translate text to '{target_language}'.")

        # Simulate translation using the static map
        translated_words = []
        # Simple tokenization by splitting on spaces. A real translator would use NLP tokenization.
        for word in data.lower().split():
            # Attempt to translate the word. If not found or language not supported for word, keep original.
            translated_word = _SIMULATED_TRANSLATION_MAP.get(word, {}).get(target_language, word)
            translated_words.append(translated_word)

        translated_text = ' '.join(translated_words)

        logger.info(
            f"[{self.node_name}] Successfully processed data. "
            f"Original (truncated): '{data[:50]}...' "
            f"Translated (truncated): '{translated_text[:50]}...'"
        )
        return translated_text

# Example usage (for testing purposes, not part of the node itself)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    translator_node = LanguageTranslatorNode()

    # Test case 1: Valid translation
    text_to_translate = "Hello world, Vishustra is a great framework example node."
    context_es = {"target_language": "es"}
    context_fr = {"target_language": "fr"}
    context_de = {"target_language": "de"}

    try:
        translated_es = translator_node.process(text_to_translate, context_es)
        print(f"Original: {text_to_translate}\nSpanish:  {translated_es}\n")

        translated_fr = translator_node.process(text_to_translate, context_fr)
        print(f"Original: {text_to_translate}\nFrench:   {translated_fr}\n")

        translated_de = translator_node.process(text_to_translate, context_de)
        print(f"Original: {text_to_translate}\nGerman:   {translated_de}\n")
    except (ValueError, KeyError) as e:
        print(f"Error during valid translation test: {e}")

    # Test case 2: Missing target_language in context
    print("\n--- Testing missing target_language ---")
    context_no_lang = {}
    try:
        translator_node.process("This should fail.", context_no_lang)
    except KeyError as e:
        print(f"Caught expected error: {e}")

    # Test case 3: Invalid input data type
    print("\n--- Testing invalid input data type ---")
    invalid_data = 123
    try:
        translator_node.process(invalid_data, context_es)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # Test case 4: Unsupported target_language (for simulated map)
    print("\n--- Testing unsupported target_language (will return original) ---")
    context_unsupported_lang = {"target_language": "jp"} # Our map doesn't have 'jp'
    text_unsupported = "hello world example"
    translated_unsupported = translator_node.process(text_unsupported, context_unsupported_lang)
    print(f"Original: {text_unsupported}\nJapanese (simulated): {translated_unsupported}\n")
    if translated_unsupported == text_unsupported:
        print("Confirmed: Returned original text as expected for unsupported language.")