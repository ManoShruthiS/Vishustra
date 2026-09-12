import logging
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra processing node that translates text from one language to another.

    This node expects the input `data` to be a string and the `context` to contain
    a 'target_language' key specifying the desired output language (e.g., 'es', 'fr', 'de').

    For demonstration purposes, this node uses a simplified, mock translation mechanism
    and does not perform actual complex language translation. It translates text word-by-word
    based on an internal dictionary.
    """

    def __init__(self, initial_translations: Optional[Dict[str, Dict[str, str]]] = None):
        """
        Initializes the LanguageTranslatorNode with an optional set of mock translations.

        Args:
            initial_translations: An optional dictionary mapping source words (lowercase) to another dictionary
                                  of target languages (e.g., 'es', 'fr') and their translations.
                                  Example: {'hello': {'es': 'hola', 'fr': 'bonjour'}}
                                  If None, a default set of mock translations will be used.
        """
        self._translations: Dict[str, Dict[str, str]] = initial_translations if initial_translations is not None else {
            "hello": {"es": "hola", "fr": "bonjour", "de": "hallo", "it": "ciao"},
            "world": {"es": "mundo", "fr": "monde", "de": "welt", "it": "mondo"},
            "good": {"es": "bueno", "fr": "bon", "de": "gut", "it": "buono"},
            "morning": {"es": "mañana", "fr": "matin", "de": "morgen", "it": "mattina"},
            "evening": {"es": "tarde", "fr": "soir", "de": "abend", "it": "sera"},
            "how": {"es": "cómo", "fr": "comment", "de": "wie", "it": "come"},
            "are": {"es": "estás", "fr": "êtes", "de": "sind", "it": "siete"},
            "you": {"es": "usted", "fr": "vous", "de": "sie", "it": "tu"},
            "this": {"es": "esto", "fr": "ceci", "de": "dies", "it": "questo"},
            "is": {"es": "es", "fr": "est", "de": "ist", "it": "è"},
            "a": {"es": "un", "fr": "un", "de": "ein", "it": "un"},
            "test": {"es": "prueba", "fr": "test", "de": "test", "it": "prova"},
        }
        logger.info(f"Initialized LanguageTranslatorNode with {len(self._translations)} base word translations.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "LanguageTranslatorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by attempting to translate it to the target language
        specified in the context.

        The translation is performed word-by-word. Words for which no translation
        is found in the target language will remain in their original form.

        Args:
            data: The input text to be translated (expected to be a string).
            context: A dictionary containing additional information, expected to have
                     a 'target_language' (str) key indicating the desired language.
                     Example: {'target_language': 'es'}

        Returns:
            The translated text as a string. If the input `data` is not a string,
            or if 'target_language' is missing from the `context`, the original
            `data` will be returned, and a warning will be logged.
        """
        if not isinstance(data, str):
            logger.warning(
                f"{self.node_name}: Input data is not a string. Expected string for translation. "
                f"Received type: {type(data).__name__}. Returning original data."
            )
            return data

        target_language = context.get("target_language")
        if not isinstance(target_language, str) or not target_language:
            logger.warning(
                f"{self.node_name}: 'target_language' not found or is not a valid string in context. "
                f"Context received: {context}. Returning original data."
            )
            return data

        # Simple word splitting for mock translation; a real translator would use NLP tokenization.
        words = data.lower().split()
        translated_words = []
        original_untranslated_count = 0

        for word in words:
            # Attempt to find a translation for the word in the specified target language
            translation = self._translations.get(word, {}).get(target_language)
            if translation:
                translated_words.append(translation)
            else:
                # If no translation is found, append the original word
                translated_words.append(word)
                original_untranslated_count += 1
                logger.debug(
                    f"{self.node_name}: No translation found for word '{word}' to '{target_language}'. "
                    f"Keeping original word."
                )

        translated_text = " ".join(translated_words)

        if original_untranslated_count > 0:
            logger.warning(
                f"{self.node_name}: {original_untranslated_count} words could not be translated "
                f"to '{target_language}' in the input. Result: '{translated_text[:100]}...'"
            )
        else:
            logger.info(
                f"{self.node_name}: Successfully translated text to '{target_language}'. "
                f"Original: '{data[:50]}...'. Translated: '{translated_text[:50]}...'"
            )

        return translated_text