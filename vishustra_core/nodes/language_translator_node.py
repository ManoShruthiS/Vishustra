import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra node responsible for simulating language translation of text data.

    This node expects the input 'data' to be a string representing the text
    to be translated. The 'context' dictionary must provide the
    'source_language' and 'target_language' as strings for the translation operation.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "LanguageTranslatorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Simulates the translation of input text from a specified source language
        to a target language.

        In a production environment, this method would integrate with an actual
        language translation service (e.g., Google Translate API, DeepL, etc.).
        For this simulation, it appends a descriptive tag to the input text.

        Args:
            data: The text string to be translated. Must be a string.
            context: A dictionary containing necessary parameters for translation.
                     Requires:
                     - 'source_language' (str): The language of the input text (e.g., "en", "fr").
                     - 'target_language' (str): The desired language for the translated text (e.g., "es", "de").

        Returns:
            A string representing the simulated translated text.

        Raises:
            ValueError: If 'data' is not a string, or if 'source_language' or
                        'target_language' are missing or invalid in the 'context'.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', but received '{type(data).__name__}'."
            )
            raise ValueError(f"Input data for {self.node_name} must be a string.")

        source_language = context.get("source_language")
        target_language = context.get("target_language")

        if not (isinstance(source_language, str) and source_language):
            logger.error(
                f"[{self.node_name}] Missing or invalid 'source_language' in context. Expected a non-empty string."
            )
            raise ValueError(f"Context for {self.node_name} requires a valid 'source_language' string.")

        if not (isinstance(target_language, str) and target_language):
            logger.error(
                f"[{self.node_name}] Missing or invalid 'target_language' in context. Expected a non-empty string."
            )
            raise ValueError(f"Context for {self.node_name} requires a valid 'target_language' string.")

        logger.info(
            f"[{self.node_name}] Initiating simulated translation from '{source_language}' to '{target_language}'."
        )

        # Simulate the translation process. In a real application, an external API call
        # would be made here to a translation service.
        translated_text = (
            f"{data} [SIMULATED_TRANSLATION from {source_language.upper()} to {target_language.upper()}]"
        )

        logger.info(
            f"[{self.node_name}] Successfully completed simulated translation to '{target_language}'."
        )

        return translated_text