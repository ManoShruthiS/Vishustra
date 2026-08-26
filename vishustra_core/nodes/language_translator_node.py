import logging
from typing import Any, Dict

# Assuming vishustra_core.nodes.base_node exists in the Python path
# and contains the BaseNode definition as provided in the project context.
from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for this module
logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra processing node that simulates translating text from a source
    language to a target language.

    This node is designed to accept a string as input `data` and requires
    the `context` dictionary to specify the `target_language` using its
    standard language code (e.g., 'en', 'es', 'fr', 'de').

    The translation mechanism is simulated by appending a descriptive tag
    to the input text, indicating the intended target language.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the unique, descriptive name of this processing node.
        """
        return "LanguageTranslator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by attempting to "translate" it to the
        language specified in the context.

        Args:
            data: The input text intended for translation. Must be a string.
            context: A dictionary providing operational parameters. It
                     **must** contain a 'target_language' key with a string
                     value representing the ISO language code for the target.
                     An optional 'source_language' key can be provided,
                     though it's not explicitly used in this simulated node.

        Returns:
            A string representing the simulated translated text.

        Raises:
            ValueError: If the input `data` is not a string, or if
                        'target_language' is missing from `context` or is
                        not a valid non-empty string.
            RuntimeError: If an unexpected issue occurs during the
                          simulated translation process.
        """
        logger.debug(f"[{self.node_name}] Processing initiated with data type: {type(data).__name__}.")

        # --- Input Data Validation ---
        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Invalid input data type. "
                f"Expected 'str', but received '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not data.strip():
            logger.warning(f"[{self.node_name}] Received empty or whitespace-only string for translation. Returning as is.")
            return data # Or raise ValueError, depending on desired behavior for empty input.

        # --- Context Validation ---
        target_language = context.get("target_language")
        if not isinstance(target_language, str) or not target_language.strip():
            error_msg = (
                f"[{self.node_name}] Context is missing 'target_language' "
                f"or it's not a valid non-empty string. "
                f"Received: '{target_language}'."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Source language is optional for this simulation
        source_language = context.get("source_language", "auto-detected")
        logger.info(
            f"[{self.node_name}] Simulating translation of text "
            f"(first 75 chars: '{data[:75]}...') "
            f"from '{source_language}' to '{target_language.upper()}'."
        )

        try:
            # --- Simulated Translation Logic ---
            # In a production environment, this block would typically involve:
            # 1. Calling an external Machine Translation API (e.g., DeepL, Google Cloud Translate).
            # 2. Invoking an internal ML model for translation.
            # 3. Handling API keys, rate limits, and language pair availability.
            # For this exercise, we simulate by appending a tag.
            translated_text = f"{data} [Translated to {target_language.upper()}]"

            logger.info(
                f"[{self.node_name}] Simulated translation completed. "
                f"Result (first 75 chars): '{translated_text[:75]}...'."
            )
            return translated_text

        except Exception as e:
            # Catching broader exceptions ensures robustness against unforeseen issues
            # from external services or complex internal logic in a real-world scenario.
            error_msg = (
                f"[{self.node_name}] An unexpected error occurred during "
                f"the translation process: {e}"
            )
            logger.exception(error_msg)  # Logs the exception with traceback
            raise RuntimeError(error_msg) from e # Re-raise for upstream handling