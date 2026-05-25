import logging
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra node designed to translate text from a source language to a target language.
    This node simulates the translation process using internal mock data.
    In a production environment, this would integrate with an external translation API
    (e.g., Google Translate, DeepL, Microsoft Translator) or an in-house model.
    """

    def __init__(self) -> None:
        """
        Initializes the LanguageTranslatorNode with mock translation data.
        """
        # A simple internal dictionary to simulate translation for demonstration purposes.
        # This allows for deterministic behavior during testing without external dependencies.
        self._mock_translations = {
            ("Hello world!", "fr"): "Bonjour le monde !",
            ("Hello world!", "es"): "¡Hola mundo!",
            ("The quick brown fox jumps over the lazy dog.", "fr"): "Le rapide renard brun saute par-dessus le chien paresseux.",
            ("The quick brown fox jumps over the lazy dog.", "es"): "El rápido zorro marrón salta sobre el perro perezoso.",
            ("Good morning", "fr"): "Bonjour",
            ("Good morning", "es"): "Buenos días",
        }
        logger.debug(f"[{self.node_name}] Initialized with mock translation data.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "LanguageTranslatorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Translates the input text based on the specified target language in the context.

        Args:
            data (Any): The text to be translated, expected as a string.
            context (Dict[str, Any]): A dictionary containing processing parameters,
                                       e.g., {"target_language": "fr", "source_language": "en"}.
                                       `target_language` is mandatory. `source_language` is optional.

        Returns:
            Any: The translated text as a string, or the original text if translation
                 is not available or if an error occurs and fallback is enabled.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If `target_language` is missing or invalid in the `context`.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', received '{type(data).__name__}'."
            )
            raise TypeError(
                f"[{self.node_name}] Input data must be a string for translation. Got '{type(data).__name__}'."
            )

        text_to_translate = data.strip()
        if not text_to_translate:
            logger.warning(
                f"[{self.node_name}] Received an empty string for translation. Returning an empty string."
            )
            return ""

        target_language = context.get("target_language")
        if not target_language or not isinstance(target_language, str):
            logger.error(
                f"[{self.node_name}] Missing or invalid 'target_language' in context. Expected a non-empty string."
            )
            raise ValueError(
                f"[{self.node_name}] 'target_language' must be provided as a non-empty string in the context."
            )
        target_language = target_language.lower() # Normalize language codes

        source_language = context.get("source_language", "en") # Default to English if not explicitly provided
        if not isinstance(source_language, str):
            logger.warning(
                f"[{self.node_name}] Invalid 'source_language' in context. Expected a string, got '{type(source_language).__name__}'. Defaulting to 'en'."
            )
            source_language = "en"
        source_language = source_language.lower() # Normalize language codes

        try:
            # Simulate actual translation using the mock dictionary.
            # In a real-world scenario, this would involve an API call to a translation service.
            translation_key = (text_to_translate, target_language)
            translated_text = self._mock_translations.get(translation_key)

            if translated_text is None:
                logger.warning(
                    f"[{self.node_name}] No mock translation found for text (from '{source_language}' to '{target_language}'): "
                    f"'{text_to_translate[:75]}{'...' if len(text_to_translate) > 75 else ''}'. "
                    "Returning original text as a fallback."
                )
                # In a production system, this could be a call to a real service,
                # or a more explicit error/fallback strategy.
                return text_to_translate

            logger.info(
                f"[{self.node_name}] Successfully translated text "
                f"(from '{source_language}' to '{target_language}'). "
                f"Input: '{text_to_translate[:50]}...'. Output: '{translated_text[:50]}...'."
            )
            return translated_text

        except Exception as e:
            logger.exception(
                f"[{self.node_name}] An unexpected error occurred during translation of "
                f"'{text_to_translate[:75]}{'...' if len(text_to_translate) > 75 else ''}' "
                f"to '{target_language}': {e}"
            )
            # Depending on requirements, one might re-raise, return original, or a specific error object.
            raise # Re-raise for robustness in a framework context.