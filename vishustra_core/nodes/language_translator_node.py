import logging
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra processing node responsible for translating text from one language
    to another. This node expects text data and a target language in the context.
    It simulates an external translation service, making it easy to integrate
    with real-world APIs later.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "LanguageTranslatorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Translates the input text data to the specified target language.

        Expected `data`: A string containing the text to be translated.
        Expected `context`: A dictionary containing:
            - 'target_language' (str): The ISO 639-1 code of the language to translate to
              (e.g., 'es' for Spanish, 'fr' for French, 'de' for German).
            - 'source_language' (Optional[str]): The ISO 639-1 code of the source language.
              If not provided, the simulated translator might attempt to detect it or
              default to a configured source language.

        Args:
            data (Any): The input data, which must be a string.
            context (Dict[str, Any]): A dictionary containing contextual information,
                                      critically including 'target_language'.

        Returns:
            Any: The translated string.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If 'target_language' is missing, empty, or not a string in the `context`.
            RuntimeError: If an unexpected error occurs during the simulated translation process.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected string, "
                f"but received {type(data).__name__}. Data: {data!r}"
            )
            raise TypeError(
                f"[{self.node_name}] Input 'data' must be a string, "
                f"but received {type(data).__name__}."
            )

        target_language: Optional[str] = context.get("target_language")
        source_language: Optional[str] = context.get("source_language")

        if not target_language or not isinstance(target_language, str):
            logger.error(
                f"[{self.node_name}] 'target_language' is missing, empty, or not a string in context: {context}. "
                "Translation cannot proceed. "
                "Please ensure 'target_language' is provided (e.g., 'es', 'fr')."
            )
            raise ValueError(
                f"[{self.node_name}] 'target_language' must be provided as a non-empty string in the context."
            )

        original_text = data
        log_snippet_original = original_text[:100] + ('...' if len(original_text) > 100 else '')
        logger.info(
            f"[{self.node_name}] Attempting to translate text from "
            f"{source_language if source_language else 'auto-detect'} to '{target_language}'. "
            f"Original snippet: '{log_snippet_original}'"
        )

        try:
            # --- Simulated Translation Service Call ---
            # In a real-world Vishustra deployment, this would be an actual call to an
            # external translation API (e.g., Google Translate, DeepL, AWS Translate)
            # using a dedicated client or SDK.
            translated_text = self._simulate_translation(original_text, target_language, source_language)
            # --- End Simulated Translation Service Call ---

            log_snippet_translated = translated_text[:100] + ('...' if len(translated_text) > 100 else '')
            logger.info(
                f"[{self.node_name}] Successfully translated text to '{target_language}'. "
                f"Translated snippet: '{log_snippet_translated}'"
            )
            return translated_text
        except Exception as e:
            logger.exception(f"[{self.node_name}] An error occurred during the translation process.")
            raise RuntimeError(f"[{self.node_name}] Failed to translate text: {e}") from e

    def _simulate_translation(self, text: str, target_language: str, source_language: Optional[str]) -> str:
        """
        Simulates an external translation service call.

        This private helper method abstracts away the specifics of an external API
        interaction. In a production environment, this would contain the actual
        client calls, retry logic, and error handling for a chosen translation provider.

        Args:
            text (str): The text to translate.
            target_language (str): The ISO 639-1 code of the language to translate the text into.
            source_language (Optional[str]): The ISO 639-1 code of the source language of the text.

        Returns:
            str: The simulated translated text. For demonstration, it provides
                 specific translations for a few common phrases and a generic
                 tagging for others.

        Raises:
            Exception: This could be raised if a real external service encounters an issue.
                       In this simulation, we'll avoid explicit raises unless needed for testing error paths.
        """
        # A simple dictionary for a few hardcoded translations for common phrases
        translation_map = {
            "hello world": {"es": "Hola Mundo", "fr": "Bonjour le monde", "de": "Hallo Welt"},
            "goodbye": {"es": "Adiós", "fr": "Au revoir", "de": "Auf Wiedersehen"},
            "thank you": {"es": "Gracias", "fr": "Merci", "de": "Danke"},
            "please": {"es": "Por favor", "fr": "S'il vous plaît", "de": "Bitte"}
        }

        # Normalize text for lookup
        normalized_text = text.lower().strip()

        # Attempt to find a specific translation
        if normalized_text in translation_map and target_language in translation_map[normalized_text]:
            simulated_translation = translation_map[normalized_text][target_language]
            logger.debug(
                f"[{self.node_name}] Used specific simulated translation for '{text}' "
                f"to '{target_language}': '{simulated_translation}'"
            )
            return simulated_translation

        # Fallback to a generic simulation if no specific translation is found
        detected_source = source_language if source_language else "auto-detected"
        simulated_translation = f"{text} [translated to {target_language} from {detected_source}]"
        logger.debug(
            f"[{self.node_name}] No specific simulated translation found for '{text}' to '{target_language}'. "
            f"Using generic simulation: '{simulated_translation}'"
        )
        return simulated_translation