import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra node designed to simulate language translation of input text.

    This node expects the input `data` to be a string. The `context` dictionary
    must contain a 'target_language' key, specifying the language to translate
    the text into (e.g., 'es' for Spanish, 'fr' for French, 'de' for German).
    An optional 'source_language' can also be provided in the context; otherwise,
    it defaults to 'auto-detection'.

    For a production environment, this node would integrate with an actual
    translation service or a robust internal translation model.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "LanguageTranslatorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Translates the provided input text data to the specified target language.

        The translation mechanism in this implementation is simulated. In a real-world
        scenario, this method would interface with an external translation API
        (e.g., Google Cloud Translation, DeepL, Microsoft Translator) or an
        in-house machine learning translation model.

        Args:
            data (Any): The input data to be translated. This is expected to be a string.
            context (Dict[str, Any]): A dictionary providing additional processing parameters.
                                      It **must** contain:
                                      - 'target_language' (str): The ISO 639-1 code of the
                                        language to translate into (e.g., "es", "fr").
                                      It **may** contain:
                                      - 'source_language' (str, optional): The ISO 639-1 code of the
                                        source language. Defaults to "auto" for auto-detection.

        Returns:
            Any: The translated text as a string.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If 'target_language' is missing from `context` or is not a valid string.
            RuntimeError: If an unexpected error occurs during the simulated translation process.
        """
        logger.debug(f"[{self.node_name}] Initiating process with data type: {type(data)}")

        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', received '{type(data).__name__}'."
            )
            raise TypeError(
                f"LanguageTranslatorNode requires string input for 'data', but received {type(data).__name__}."
            )

        target_language = context.get("target_language")
        if not isinstance(target_language, str) or not target_language.strip():
            logger.error(
                f"[{self.node_name}] 'target_language' is missing or invalid in context. "
                f"Received keys: {list(context.keys())}"
            )
            raise ValueError(
                "Context must contain a valid non-empty string for 'target_language' for translation."
            )

        source_language = context.get("source_language", "auto").strip().lower()
        effective_target_language = target_language.strip().lower()

        logger.info(
            f"[{self.node_name}] Attempting to translate text "
            f"from '{source_language}' to '{effective_target_language}'. "
            f"Input snippet: '{data[:75]}{'...' if len(data) > 75 else ''}'"
        )

        try:
            # --- Simulated Translation Logic ---
            # This block simulates the interaction with an external translation service.
            # In a real application, this would involve API calls, error handling for
            # external service responses, and potentially retry mechanisms.
            if effective_target_language == "es":
                translated_text = f"[ES] {data} [Traducido al español]"
            elif effective_target_language == "fr":
                translated_text = f"[FR] {data} [Traduit en français]"
            elif effective_target_language == "de":
                translated_text = f"[DE] {data} [Übersetzt ins Deutsche]"
            elif effective_target_language == "ja":
                translated_text = f"[JA] {data} [日本語に翻訳済み]"
            elif effective_target_language == "zh":
                translated_text = f"[ZH] {data} [翻译成中文]"
            else:
                # For any other unsupported or generic language, append a generic translated tag.
                translated_text = f"[{effective_target_language.upper()}] {data} [Translated]"
            # --- End Simulated Translation Logic ---

            logger.info(
                f"[{self.node_name}] Successfully translated text to '{effective_target_language}'. "
                f"Output snippet: '{translated_text[:75]}{'...' if len(translated_text) > 75 else ''}'"
            )
            return translated_text

        except Exception as e:
            logger.exception(
                f"[{self.node_name}] An unexpected error occurred during the simulated translation process."
            )
            raise RuntimeError(f"Failed to perform language translation: {e}") from e