import logging
from typing import Any, Dict

# Assuming vishustra_core is a package and nodes.base_node is a module within it
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra processing node that simulates language translation.

    This node takes a text string and a target language from the context,
    and returns a simulated translated string.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "LanguageTranslator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by simulating translation to a target language.

        Expected `data`:
            - A string containing the text to be translated.

        Expected `context` keys:
            - 'target_language' (str): The ISO 639-1 code or full name of the
              language to translate the text into (e.g., 'es', 'fr', 'German').
              This key is mandatory.
            - 'source_language' (str, optional): The ISO 639-1 code or full name
              of the source language. If not provided, it's assumed to be
              auto-detected by an underlying translation service.

        Returns:
            - A string representing the simulated translated text.

        Raises:
            - TypeError: If `data` is not a string.
            - ValueError: If 'target_language' is missing or invalid in the `context`.
            - RuntimeError: For any unexpected errors during the simulated translation process.
        """
        logger.debug(f"[{self.node_name}] Starting process for data type: {type(data)}")
        logger.debug(f"[{self.node_name}] Context received: {context}")

        if not isinstance(data, str):
            logger.error(f"[{self.node_name}] Invalid input data type. Expected string, got {type(data)}.")
            raise TypeError(
                f"[{self.node_name}] Invalid input data. Expected a string for translation, but received {type(data)}."
            )

        target_language = context.get("target_language")
        source_language = context.get("source_language", "auto-detected")

        if not target_language or not isinstance(target_language, str):
            logger.error(
                f"[{self.node_name}] 'target_language' is missing or invalid in context. "
                f"Received: '{target_language}' (type: {type(target_language)})"
            )
            raise ValueError(
                f"[{self.node_name}] 'target_language' must be provided as a non-empty string in the context."
            )
        
        # Log the attempt, truncating long data for readability
        data_preview = data[:100] + ('...' if len(data) > 100 else '')
        logger.info(
            f"[{self.node_name}] Attempting to translate from '{source_language}' "
            f"to '{target_language}' for text: '{data_preview}'"
        )

        try:
            # Simulate translation. In a real scenario, this would involve
            # calling an external translation API (e.g., Google Translate, DeepL).
            # For this simulation, we append a clear indicator to the original text.
            translated_text = f"{data} [Translated to {target_language.upper()}]"
            
            # Log the simulated result, again truncating if necessary
            translated_preview = translated_text[:100] + ('...' if len(translated_text) > 100 else '')
            logger.info(
                f"[{self.node_name}] Successfully simulated translation to: "
                f"'{translated_preview}'"
            )
            return translated_text
        except Exception as e:
            # Catch any unexpected errors that might occur during the (simulated) translation logic
            logger.exception(f"[{self.node_name}] An unexpected error occurred during translation simulation.")
            raise RuntimeError(f"[{self.node_name}] Failed to process translation due to an internal error: {e}") from e