import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class LanguageTranslator(BaseNode):
    """
    A Vishustra node that simulates language translation of text data.

    This node expects the input 'data' to be a string (the text to translate)
    and the 'context' dictionary to contain a 'target_language' key,
    specifying the language code (e.g., 'es' for Spanish, 'fr' for French).
    An optional 'source_language' key can also be provided.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "LanguageTranslator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input text data by simulating translation to a target language.

        Args:
            data (Any): The input data, expected to be a string for translation.
            context (Dict[str, Any]): A dictionary containing processing context.
                                      Must include 'target_language' (str).
                                      Can optionally include 'source_language' (str).

        Returns:
            Any: The simulated translated text (string).

        Raises:
            TypeError: If 'data' is not a string.
            ValueError: If 'target_language' is missing from 'context' or is not a string.
        """
        logger.debug(
            f"[{self.node_name}] Starting process for data (first 50 chars): {str(data)[:50]}..."
        )

        # --- Input validation for data ---
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', got '{type(data).__name__}'."
            )
            raise TypeError(
                f"[{self.node_name}] Input 'data' must be a string for translation. "
                f"Got type: {type(data).__name__}."
            )

        # --- Context validation for target_language ---
        target_language = context.get("target_language")
        if not target_language or not isinstance(target_language, str):
            logger.error(
                f"[{self.node_name}] Missing or invalid 'target_language' in context. "
                f"Context provided keys: {list(context.keys())}"
            )
            raise ValueError(
                f"[{self.node_name}] 'target_language' (string) must be provided in the context for translation."
            )

        source_language = context.get("source_language", "auto")
        logger.info(
            f"[{self.node_name}] Attempting to translate from '{source_language}' to '{target_language}'."
        )

        # --- Simulate translation ---
        # In a real-world scenario, this section would involve an API call to
        # an external translation service (e.g., Google Translate, DeepL, Azure Translator).
        # For this simulation, we'll append a descriptive string indicating the target language.
        simulated_translations_map = {
            "en": "[translated to English]",
            "es": "[traducido al español]",
            "fr": "[traduit en français]",
            "de": "[ins Deutsche übersetzt]",
            "zh": "[翻译成中文]",
            "ja": "[日本語に翻訳されました]",
            "hi": "[हिंदी में अनुवादित]",
        }

        # Normalize target language for lookup (e.g., 'EN' -> 'en')
        normalized_target_lang = target_language.lower()
        translation_suffix = simulated_translations_map.get(normalized_target_lang)

        translated_data = data
        if translation_suffix:
            translated_data = f"{data} {translation_suffix}"
            logger.info(
                f"[{self.node_name}] Successfully simulated translation to '{target_language}'."
            )
        else:
            logger.warning(
                f"[{self.node_name}] Unsupported target language '{target_language}' for simulation. "
                f"Returning original data without modification."
            )

        logger.debug(
            f"[{self.node_name}] Finished process. Result (first 50 chars): {translated_data[:50]}..."
        )
        return translated_data