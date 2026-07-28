import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslator(BaseNode):
    """
    A Vishustra processing node that simulates language translation of text data.

    This node expects text data as input and a 'target_language' specified
    in the context dictionary. It provides a mocked translation for demonstration
    purposes. In a production environment, this would interface with a
    real translation API.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "LanguageTranslator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Translates the input text data to the specified target language.

        The `context` dictionary must contain a 'target_language' key
        with the ISO 639-1 code (e.g., 'fr', 'es', 'de').

        Args:
            data (Any): The text data to be translated. Expected to be a string.
                        If not a string, an attempt will be made to convert it.
            context (Dict[str, Any]): A dictionary containing operational context,
                                      including 'target_language'.

        Returns:
            Any: The translated text as a string.

        Raises:
            ValueError: If 'target_language' is missing in the context or
                        if the input data cannot be processed as text.
        """
        if not isinstance(data, str):
            logger.warning(
                f"Input data for LanguageTranslator is of type {type(data).__name__}. "
                "Attempting to convert to string for translation."
            )
            try:
                data = str(data)
            except Exception as e:
                logger.error(f"Failed to convert input data to string: {e}")
                raise ValueError(
                    "LanguageTranslator expects string data or data convertible to string."
                ) from e

        target_language = context.get("target_language")
        if not target_language or not isinstance(target_language, str):
            logger.error(
                "Missing or invalid 'target_language' in context for LanguageTranslator. "
                "Expected a string (e.g., 'fr', 'es')."
            )
            raise ValueError(
                "Translation requires a valid 'target_language' (str) in context."
            )

        logger.info(
            f"Attempting to translate data to '{target_language}': '{data[:100]}{'...' if len(data) > 100 else ''}'"
        )

        # --- Simulated Translation Logic ---
        # In a real scenario, this would involve calling a translation API
        # (e.g., Google Translate, DeepL, Azure Translator).
        # For demonstration, we use a simple mock dictionary and word-by-word substitution.
        
        mock_phrase_translations = {
            "hello world": {"fr": "bonjour le monde", "es": "hola mundo", "de": "hallo welt"},
            "how are you": {"fr": "comment allez-vous", "es": "¿cómo estás?", "de": "wie geht es dir"},
            "thank you": {"fr": "merci", "es": "gracias", "de": "danke schön"},
            "please translate this": {"fr": "veuillez traduire ceci", "es": "por favor traduce esto", "de": "bitte übersetzen sie dies"},
        }
        
        # Check for whole phrase matches first
        data_lower = data.lower()
        if data_lower in mock_phrase_translations and target_language in mock_phrase_translations[data_lower]:
            translated_data = mock_phrase_translations[data_lower][target_language]
            logger.debug(f"Translated whole phrase '{data_lower}' to '{translated_data}' using mock.")
        else:
            # Fallback to word-by-word simulation
            mock_word_translations = {
                "hello": {"fr": "bonjour", "es": "hola", "de": "hallo"},
                "world": {"fr": "monde", "es": "mundo", "de": "welt"},
                "how": {"fr": "comment", "es": "cómo", "de": "wie"},
                "are": {"fr": "êtes", "es": "estás", "de": "sind"}, # Simplified verb conjugation
                "you": {"fr": "vous", "es": "tú", "de": "du"},
                "i": {"fr": "je", "es": "yo", "de": "ich"},
                "am": {"fr": "suis", "es": "soy", "de": "bin"},
                "a": {"fr": "un", "es": "un", "de": "ein"},
                "the": {"fr": "le", "es": "el", "de": "der"},
                "this": {"fr": "ceci", "es": "esto", "de": "dies"},
                "is": {"fr": "est", "es": "es", "de": "ist"},
                "good": {"fr": "bon", "es": "bueno", "de": "gut"},
            }

            translated_parts = []
            # Split by whitespace, preserving original casing where possible for untranslated parts
            words = data.split()
            
            for word in words:
                original_word_lower = word.lower()
                if original_word_lower in mock_word_translations and target_language in mock_word_translations[original_word_lower]:
                    translated_parts.append(mock_word_translations[original_word_lower][target_language])
                else:
                    # For words not in our mock, we'll append a marker to indicate simulation
                    translated_parts.append(f"{word}[{target_language}]")
            
            translated_data = " ".join(translated_parts)
            logger.debug(f"Translated word-by-word to '{translated_data}' using mock.")

        logger.info(
            f"Translation completed to '{target_language}'. Output: '{translated_data[:100]}{'...' if len(translated_data) > 100 else ''}'"
        )
        return translated_data
