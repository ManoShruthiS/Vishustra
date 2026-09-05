import logging
from typing import Any, Dict

# Assuming BaseNode is available at this path within the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra processing node that simulates language translation of text data.

    This node expects a string as input `data` and requires 'target_language'
    in the `context` dictionary to perform its simulated translation.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "LanguageTranslator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Translates the input text data to a specified target language.

        The translation is simulated by appending a language tag to the original text.

        Args:
            data: The text data (str) to be translated.
            context: A dictionary containing operational parameters.
                     Requires 'target_language' (str) to specify the translation target.

        Returns:
            str: The simulated translated text.

        Raises:
            TypeError: If `data` is not a string or `target_language` is not a string.
            ValueError: If 'target_language' is missing from the context.
            Exception: For any other unexpected errors during processing.
        """
        node_id = context.get("node_id", self.node_name)
        logger.info(f"[{node_id}] Initiating language translation process.")

        try:
            if not isinstance(data, str):
                logger.error(
                    f"[{node_id}] Invalid input data type. Expected 'str', but got '{type(data).__name__}'."
                )
                raise TypeError("Input 'data' must be a string for translation.")

            target_language = context.get("target_language")

            if target_language is None:
                logger.error(f"[{node_id}] 'target_language' is missing in the context for translation.")
                raise ValueError("Context must contain 'target_language' for the LanguageTranslatorNode.")

            if not isinstance(target_language, str):
                logger.error(
                    f"[{node_id}] Invalid 'target_language' type. Expected 'str', but got '{type(target_language).__name__}'."
                )
                raise TypeError("The 'target_language' in context must be a string.")

            logger.debug(f"[{node_id}] Translating text (len={len(data)}) to '{target_language}'.")

            # Simulate the translation process
            # In a real scenario, this would involve an external API call (e.g., Google Translate, DeepL)
            translated_data = f"{data} [Translated to {target_language.upper()}]"

            logger.info(f"[{node_id}] Successfully simulated translation to '{target_language}'.")
            return translated_data

        except (TypeError, ValueError) as e:
            logger.exception(f"[{node_id}] Data validation error during translation: {e}")
            raise  # Re-raise the specific exception
        except Exception as e:
            logger.exception(f"[{node_id}] An unexpected error occurred during language translation: {e}")
            raise # Re-raise any other unexpected exceptions