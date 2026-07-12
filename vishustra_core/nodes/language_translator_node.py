import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra processing node that simulates language translation of text data.

    This node expects a string as input data and a 'target_language' key
    in the context dictionary to determine the language for translation.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "LanguageTranslator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by simulating translation to a target language.

        Args:
            data (Any): The input data, expected to be a string for translation.
            context (Dict[str, Any]): A dictionary containing contextual information.
                                      Must include 'target_language' (str)
                                      specifying the language to translate to (e.g., 'es', 'fr', 'de').

        Returns:
            Any: The translated string.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If 'target_language' is missing in context or is unsupported.
        """
        if not isinstance(data, str):
            logger.error("LanguageTranslatorNode received non-string data. Type: %s", type(data).__name__)
            raise TypeError(f"LanguageTranslatorNode expects string data, but received {type(data).__name__}.")

        target_language_raw = context.get("target_language")
        if not target_language_raw:
            logger.error("LanguageTranslatorNode context missing 'target_language' key.")
            raise ValueError("Context must include 'target_language' for LanguageTranslatorNode.")
        
        target_language = str(target_language_raw).lower() # Normalize target language string

        logger.info("Attempting to translate data (first 50 chars): '%s' to '%s'.", 
                    data[:50] + ("..." if len(data) > 50 else ""), target_language)

        translated_data: str
        if target_language == "es":
            translated_data = f"[ES] {data} [Traducido al español]"
        elif target_language == "fr":
            translated_data = f"[FR] {data} [Traduit en français]"
        elif target_language == "de":
            translated_data = f"[DE] {data} [Ins Deutsche übersetzt]"
        elif target_language == "ja":
            translated_data = f"[JA] {data} [日本語に翻訳されました]"
        elif target_language == "zh":
            translated_data = f"[ZH] {data} [翻译成中文]"
        else:
            logger.error("LanguageTranslatorNode received unsupported target_language: '%s'.", target_language)
            raise ValueError(
                f"Unsupported target_language: '{target_language}'. "
                "Supported languages for simulation include: es, fr, de, ja, zh."
            )

        logger.debug("Translation complete. Original (first 50 chars): '%s', Translated (first 50 chars): '%s'", 
                     data[:50], translated_data[:50])
        return translated_data