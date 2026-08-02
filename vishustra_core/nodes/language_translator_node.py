import logging
from typing import Any, Dict

# Assuming `vishustra_core` is installed and `base_node.py` is located under `vishustra_core/nodes/`
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A processing node designed to simulate language translation of text data.

    This node expects the input `data` to be a string representing the text
    to be translated. The `context` dictionary must contain a 'target_language'
    key, specifying the language code (e.g., 'es', 'fr', 'de') for the translation.

    In a production environment, this node would integrate with an actual
    translation service or a sophisticated LLM capable of language translation.
    For this simulation, it provides deterministic translations for a few
    predefined phrases and falls back to a tagging mechanism for others.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "LanguageTranslator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Simulates the translation of input text data to a specified target language.

        Args:
            data: The input data, expected to be a string containing the text
                  to be translated.
            context: A dictionary containing additional execution context.
                     It MUST include 'target_language' (str) indicating the
                     language to translate the `data` into.

        Returns:
            A string representing the simulated translated text. If the translation
            cannot be performed due to missing parameters or unexpected errors,
            an exception is raised.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If 'target_language' is missing from `context` or is not a string.
            RuntimeError: For any other operational errors encountered during the
                          simulated translation process.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"but received '{type(data).__name__}'. Data: {data!r}"
            )
            raise TypeError(
                f"[{self.node_name}] Input 'data' must be a string, "
                f"received '{type(data).__name__}'."
            )

        target_language = context.get("target_language")
        if not target_language or not isinstance(target_language, str):
            logger.error(
                f"[{self.node_name}] Missing or invalid 'target_language' in context. "
                f"Context received: {context!r}"
            )
            raise ValueError(
                f"[{self.node_name}] Context requires a 'target_language' (str). "
                f"Received '{target_language}'."
            )

        try:
            logger.info(
                f"[{self.node_name}] Attempting to translate text "
                f"(length: {len(data)}) to '{target_language}'."
            )

            # Simulate the translation process.
            # In a real-world application, this would involve calling a dedicated
            # translation API (e.g., from Google Cloud, AWS, DeepL) or
            # interfacing with an LLM for translation tasks.
            translated_text: str = self._perform_simulated_translation(data, target_language)

            logger.info(
                f"[{self.node_name}] Successfully translated text to '{target_language}'. "
                f"Original snippet: '{data[:50]}...'. Translated snippet: '{translated_text[:50]}...'"
            )
            return translated_text
        except Exception as e:
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during translation "
                f"to '{target_language}': {e}",
                exc_info=True
            )
            raise RuntimeError(
                f"[{self.node_name}] Translation failed for target language "
                f"'{target_language}': {e}"
            ) from e

    def _perform_simulated_translation(self, text: str, target_lang: str) -> str:
        """
        Internal helper method to simulate the actual translation logic.
        This serves as a placeholder for an external translation API call.
        """
        # A simple, deterministic dictionary for simulated translations
        _SIMULATED_TRANSLATIONS = {
            ("Hello world", "es"): "Hola mundo",
            ("Hello world", "fr"): "Bonjour le monde",
            ("Thank you", "es"): "Gracias",
            ("Thank you", "fr"): "Merci",
            ("Good morning", "de"): "Guten Morgen",
            ("This is a test", "es"): "Esto es una prueba",
            ("This is a test", "fr"): "Ceci est un test",
            ("Welcome to Vishustra", "es"): "Bienvenido a Vishustra",
            ("Welcome to Vishustra", "fr"): "Bienvenue à Vishustra",
        }

        # Check for an exact phrase match in our predefined map
        translation_key = (text, target_lang)
        if translation_key in _SIMULATED_TRANSLATIONS:
            return _SIMULATED_TRANSLATIONS[translation_key]

        # For any text not explicitly defined, we simulate by prepending a tag.
        # This highlights that a real translation service would be dynamic.
        logger.warning(
            f"[{self.node_name}] No exact simulated translation found for text "
            f"'{text[:50]}...' to '{target_lang}'. Using generic tagging."
        )
        return f"[Translated to {target_lang}]: {text}"

