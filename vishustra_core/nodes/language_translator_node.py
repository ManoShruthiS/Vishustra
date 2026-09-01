import logging
from typing import Any, Dict

# Assuming BaseNode is available at this path within the Vishustra framework
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra processing node that simulates language translation.

    This node expects the input 'data' to be a string (the text to translate)
    and the 'context' dictionary to contain a 'target_language' key with
    the desired language code (e.g., 'es', 'fr', 'de').

    It simulates translation by appending a suffix indicating the target language.
    In a production environment, this would integrate with a real translation service
    like Google Translate, DeepL, or a custom NMT model.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "LanguageTranslator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Simulates the translation of input text to a specified target language.

        Args:
            data: The input data, expected to be a string containing the text to translate.
            context: A dictionary containing runtime context for the process.
                     It MUST include 'target_language' (str) indicating the
                     desired output language code.

        Returns:
            A string representing the 'translated' text.

        Raises:
            TypeError: If 'data' is not a string.
            ValueError: If 'target_language' is missing from the context or not a string.
            Exception: For any other unexpected errors during the translation simulation.
        """
        logger.debug(f"[{self.node_name}] Initiating processing for data type: {type(data).__name__}")

        # Validate input data type
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', but received '{type(data).__name__}'."
            )
            raise TypeError(
                f"LanguageTranslatorNode expects 'data' to be a string for translation, "
                f"but received type '{type(data).__name__}'."
            )

        # Validate and retrieve target language from context
        target_language = context.get("target_language")
        if not target_language or not isinstance(target_language, str):
            logger.error(
                f"[{self.node_name}] Missing or invalid 'target_language' in context. "
                f"Expected a non-empty string, but got '{type(target_language).__name__}' "
                f"or None." if target_language else "None."
            )
            raise ValueError(
                f"LanguageTranslatorNode requires 'target_language' (str) in the context "
                f"to perform translation. Received: '{target_language}'."
            )

        try:
            # --- Simulation of translation logic ---
            # In a real-world scenario, this section would involve calling a
            # translation service API (e.g., `requests.post(translation_api_url, ...)`),
            # handling network errors, API specific responses, and rate limits.
            translated_text = f"{data} [translated to {target_language}]"
            # ---------------------------------------

            logger.info(
                f"[{self.node_name}] Successfully simulated translation of text "
                f"(first 50 chars: '{data[:50]}{'...' if len(data) > 50 else ''}') "
                f"to '{target_language}'."
            )
            return translated_text
        except Exception as e:
            logger.exception(
                f"[{self.node_name}] An unhandled error occurred during the "
                f"translation simulation for text (first 50 chars: '{data[:50]}{'...' if len(data) > 50 else ''}') "
                f"to '{target_language}'. Error: {e}"
            )
            # Re-raise the exception to propagate the failure upstream
            raise
