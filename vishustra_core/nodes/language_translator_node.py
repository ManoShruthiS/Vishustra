import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslator(BaseNode):
    """
    A Vishustra processing node that simulates language translation of text data.

    This node expects string data and a 'target_language' in the context
    to simulate the translation process. It includes robust error handling
    for invalid inputs and unsupported languages.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "LanguageTranslator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by simulating translation to a specified target language.

        Args:
            data: The input data, expected to be a string containing the text to translate.
            context: A dictionary containing operational context, expected to include
                     'target_language' (str, e.g., 'es', 'fr', 'de').

        Returns:
            A string representing the simulated translated text.

        Raises:
            TypeError: If 'data' is not a string.
            ValueError: If 'target_language' is missing from 'context' or is not a string.
            RuntimeError: If an unexpected error occurs during the simulated translation.
        """
        logger.debug(f"[{self.node_name}] Starting process for data type: {type(data)}.")

        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid data type. Expected 'str', but received '{type(data)}'."
            )
            raise TypeError(
                f"[{self.node_name}] Input 'data' must be a string for translation. Got: {type(data)}"
            )

        target_language = context.get("target_language")

        if not target_language or not isinstance(target_language, str):
            logger.error(
                f"[{self.node_name}] 'target_language' missing from context or is not a string. "
                "Context: {context}"
            )
            raise ValueError(
                f"[{self.node_name}] Context must contain a 'target_language' string "
                "(e.g., 'es', 'fr', 'de') to perform translation."
            )

        original_text: str = data
        translated_text: str = original_text # Initialize with original text as fallback

        # A simple map for simulating translations for demonstration purposes.
        # In a real scenario, this would involve an external API call or a complex model.
        translation_simulations = {
            "es": "[translated to Spanish]",
            "fr": "[traduit en français]",
            "de": "[ins Deutsche übersetzt]",
            "it": "[tradotto in italiano]",
            "ja": "[日本語に翻訳されました]",
            "zh": "[翻译成中文]",
        }

        try:
            if target_language in translation_simulations:
                translated_text = f"{original_text} {translation_simulations[target_language]}"
                logger.info(
                    f"[{self.node_name}] Successfully simulated translation to '{target_language}'. "
                    f"Original (first 50 chars): '{original_text[:50]}...'"
                )
            else:
                logger.warning(
                    f"[{self.node_name}] Unsupported target language '{target_language}'. "
                    "Returning original text without translation."
                )
                translated_text = original_text # No change for unsupported language
        except Exception as e:
            # Catching general exceptions during simulation to provide robust error handling
            logger.exception(
                f"[{self.node_name}] An unexpected error occurred during simulated translation "
                f"to '{target_language}'."
            )
            raise RuntimeError(
                f"[{self.node_name}] Translation simulation failed for target language "
                f"'{target_language}': {e}"
            ) from e

        logger.debug(
            f"[{self.node_name}] Process completed. Original data (first 50 chars): '{original_text[:50]}...'. "
            f"Translated data (first 50 chars): '{translated_text[:50]}...'"
        )
        return translated_text