import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra node responsible for simulating language translation of text data.

    This node processes an input string, attempting to translate it into a
    specified target language using parameters from the execution context.
    It is designed to integrate into orchestration pipelines where text
    localization or cross-lingual communication is required.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "LanguageTranslator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Translates the input text based on the 'target_language' provided in the context.

        Args:
            data: The text content (string) to be translated.
            context: A dictionary containing execution parameters, expected to include:
                     - 'target_language' (str): The ISO 639-1 code for the language
                                                to translate into (e.g., 'en', 'es', 'fr').
                     - 'source_language' (str, optional): The ISO 639-1 code for the original
                                                          language of the text. Defaults to 'auto-detect'.

        Returns:
            str: The simulated translated text. In a real implementation, this would be
                 the output from an external translation service.

        Raises:
            ValueError: If `data` is not a string or if 'target_language' is missing from `context`.
            RuntimeError: If an unexpected error occurs during the translation simulation.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"but received '{type(data).__name__}'."
            )
            raise ValueError(f"[{self.node_name}] Input data must be a string for translation.")

        target_language = context.get('target_language')
        if not target_language or not isinstance(target_language, str):
            logger.error(
                f"[{self.node_name}] 'target_language' key is missing or not a string "
                f"in the context. Translation cannot proceed."
            )
            raise ValueError(
                f"[{self.node_name}] 'target_language' (str) must be provided in the context "
                f"for translation."
            )

        source_language = context.get('source_language', 'auto-detect')
        logger.info(
            f"[{self.node_name}] Initiating translation from '{source_language}' "
            f"to '{target_language}' for input text (first 50 chars): '{data[:50]}...'"
        )

        try:
            # Simulate the translation process.
            # In a production environment, this would involve an API call to a
            # language translation service (e.g., Google Translate, DeepL, Azure Translator).
            translated_text = f"{data} [Translated to {target_language.upper()}]"
            logger.debug(
                f"[{self.node_name}] Successfully simulated translation. "
                f"Output (first 50 chars): '{translated_text[:50]}...'"
            )
            return translated_text
        except Exception as e:
            logger.critical(
                f"[{self.node_name}] An unexpected error occurred during language translation "
                f"simulation: {e}", exc_info=True
            )
            raise RuntimeError(f"[{self.node_name}] Failed to complete translation simulation: {e}") from e