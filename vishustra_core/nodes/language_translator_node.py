import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra processing node that simulates language translation of text data.

    This node expects the input 'data' to be a string and the 'context'
    dictionary to contain a 'target_language' key, specifying the language
    to translate the text into (e.g., 'es', 'fr', 'de', 'ja').

    For a real-world scenario, this node would integrate with a robust
    third-party translation service API. This implementation provides
    a clear simulation for demonstration and testing purposes.
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
            context: A dictionary containing processing context,
                     expected to have 'target_language' (str).

        Returns:
            The translated string, or the original data with an appended note
            if translation fails or the target language is not supported
            by this simulator.

        Raises:
            TypeError: If 'data' is not a string.
            ValueError: If 'target_language' is missing or invalid in the context.
        """
        if not isinstance(data, str):
            logger.error(
                "LanguageTranslatorNode received non-string data. "
                "Expected a string for translation. Received type: %s", type(data)
            )
            raise TypeError("LanguageTranslatorNode expects string data for translation.")

        target_language = context.get("target_language")
        if not isinstance(target_language, str) or not target_language:
            logger.error(
                "LanguageTranslatorNode requires 'target_language' (str) "
                "in context, but received: %s", target_language
            )
            raise ValueError("Context missing or invalid 'target_language' for translation.")

        # Simulate translation. In a production system, this would involve
        # an API call to a service like Google Translate, DeepL, etc.
        # For demonstration, we use a simple mapping.
        simulated_translations = {
            "es": f"¡Hola! (Translated to Spanish): '{data}'",
            "fr": f"Bonjour ! (Traduit en français): '{data}'",
            "de": f"Hallo! (Übersetzt ins Deutsche): '{data}'",
            "ja": f"こんにちは！ (日本語に翻訳): '{data}'",
            # Add more simulated translations as needed
        }

        translated_text = simulated_translations.get(target_language.lower())

        if translated_text:
            log_data_snippet = data[:75] + '...' if len(data) > 75 else data
            logger.info(
                "Successfully simulated translation to '%s' for data: '%s'",
                target_language, log_data_snippet
            )
            return translated_text
        else:
            logger.warning(
                "Unsupported target language '%s' for LanguageTranslatorNode simulator. "
                "Returning original data with a note.", target_language
            )
            return data + f" (Translation to {target_language} not supported by simulator)"