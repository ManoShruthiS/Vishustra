import logging
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra processing node that simulates language translation of text data.

    This node expects the input `data` to be a string and the `context`
    dictionary to contain a 'target_language' key specifying the language
    to translate the text into.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "LanguageTranslator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by simulating translation to a target language.

        Args:
            data (Any): The input data to be translated. Expected to be a string.
            context (Dict[str, Any]): A dictionary containing additional
                                       information for processing.
                                       Must include 'target_language' (str).

        Returns:
            Any: The simulated translated string.

        Raises:
            ValueError: If `data` is not a string, or 'target_language' is
                        missing or not a string in the context.
            RuntimeError: If a simulated translation error occurs during processing.
        """
        if not isinstance(data, str):
            error_msg = (
                f"LanguageTranslatorNode expects string input for data, "
                f"but received type: {type(data).__name__}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        target_language: Optional[str] = context.get("target_language")

        if not target_language:
            error_msg = "LanguageTranslatorNode requires 'target_language' in context for translation."
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not isinstance(target_language, str):
            error_msg = (
                f"'target_language' in context must be a string, "
                f"but received type: {type(target_language).__name__}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            translated_text = self._simulate_translation(data, target_language)
            logger.debug(
                f"Successfully simulated translation of text (first 50 chars: '{data[:50]}...') "
                f"to target_language='{target_language}'. Result: '{translated_text[:50]}...'"
            )
            return translated_text
        except Exception as e:
            error_msg = (
                f"Failed to simulate translation for data (first 50 chars: '{data[:50]}...') "
                f"to language '{target_language}'. Error: {e}"
            )
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def _simulate_translation(self, text: str, target_language: str) -> str:
        """
        Internal method to simulate the translation process.

        This method provides a highly simplified, mock translation for demonstration.
        In a production environment, this would integrate with a robust external
        translation service (e.g., Google Translate API, DeepL, etc.) or an
        internal machine learning model.
        """
        # A very simplified mapping for common words/phrases for demonstration.
        mock_translations = {
            "en": {
                "hello": "hello",
                "world": "world",
                "data processing": "data processing",
                "vishustra": "vishustra",
                "node": "node",
                "framework": "framework",
            },
            "fr": {
                "hello": "bonjour",
                "world": "monde",
                "data processing": "traitement de données",
                "vishustra": "vishustra",
                "node": "nœud",
                "framework": "cadre",
            },
            "es": {
                "hello": "hola",
                "world": "mundo",
                "data processing": "procesamiento de datos",
                "vishustra": "vishustra",
                "node": "nodo",
                "framework": "marco",
            },
            "de": {
                "hello": "hallo",
                "world": "welt",
                "data processing": "datenverarbeitung",
                "vishustra": "vishustra",
                "node": "knoten",
                "framework": "rahmen",
            },
        }

        target_language_lower = target_language.lower()

        if target_language_lower not in mock_translations:
            logger.warning(
                f"Unsupported target language '{target_language}' for simulated translation. "
                "Returning original text with language tag prefix."
            )
            # Fallback for languages not in our mock dictionary
            return f"[{target_language.upper()}] {text}"

        translated_words = []
        # Simple word-by-word translation. This is highly simplistic and for simulation only.
        # Real translators handle context, grammar, morphology, etc., which is beyond this scope.
        for word in text.split():
            # Basic lowercasing for dictionary lookup.
            # Attempt to preserve original capitalization for the first letter if a translation exists.
            lower_word = word.lower()
            translated_candidate = mock_translations[target_language_lower].get(
                lower_word
            )

            if translated_candidate:
                # Restore original capitalization if the original word was capitalized
                if word and word[0].isupper():
                    translated_candidate = translated_candidate.capitalize()
                translated_words.append(translated_candidate)
            else:
                translated_words.append(word)  # If no translation, keep original word

        translated_sentence = " ".join(translated_words)
        return f"[{target_language.upper()}] {translated_sentence}"