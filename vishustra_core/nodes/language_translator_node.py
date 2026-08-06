import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra node designed to simulate language translation of text data.

    This node processes an input string, simulating its translation based on
    a specified target language provided in the context. It's built to be
    integrated into larger orchestration workflows where text localization
    or cross-language processing is required.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "LanguageTranslatorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Simulates the translation of input text data to a specified target language.

        For this implementation, translation is simulated by appending a tag
        indicating the target language. In a production environment, this method
        would integrate with an actual language translation service or model.

        Args:
            data: The input data, expected to be a string representing the text
                  to be translated.
            context: A dictionary containing contextual information necessary for
                     processing. It *must* include the 'target_language' key,
                     whose value should be a string indicating the desired
                     target language (e.g., "French", "Spanish", "German").

        Returns:
            A string representing the simulated translated text. If the input `data`
            is an empty or whitespace-only string, an empty string is returned.

        Raises:
            TypeError: If `data` is not a string, or if 'target_language' in the
                       context is not a string.
            ValueError: If 'target_language' is missing from the `context` dictionary.
        """
        logger.debug(f"[{self.node_name}] Initiating text translation process for input data.")

        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"but received '{type(data).__name__}'. Aborting translation."
            )
            raise TypeError(
                f"[{self.node_name}] Input data for translation must be a string. "
                f"Received type: {type(data).__name__}."
            )

        if not data.strip():
            logger.warning(
                f"[{self.node_name}] Received an empty or whitespace-only string for translation. "
                f"Returning an empty string as result."
            )
            return ""

        target_language = context.get("target_language")

        if target_language is None:
            logger.error(
                f"[{self.node_name}] 'target_language' key is missing from the context. "
                f"Cannot proceed with translation without a specified target language."
            )
            raise ValueError(
                f"[{self.node_name}] 'target_language' must be provided in the context "
                f"for the LanguageTranslatorNode to function."
            )

        if not isinstance(target_language, str):
            logger.error(
                f"[{self.node_name}] Invalid type for 'target_language' in context. Expected 'str', "
                f"but received '{type(target_language).__name__}'. Aborting translation."
            )
            raise TypeError(
                f"[{self.node_name}] 'target_language' in context must be a string. "
                f"Received type: {type(target_language).__name__}."
            )

        # Simulate the translation process. In a real scenario, this would involve
        # an API call to a translation service (e.g., Google Translate, DeepL)
        # or an interaction with an in-house NLP model.
        translated_text = f"{data} [Translated to {target_language}]"
        logger.info(
            f"[{self.node_name}] Successfully simulated translation to '{target_language}'. "
            f"Original (truncated): '{data[:70]}{'...' if len(data) > 70 else ''}'. "
            f"Translated (truncated): '{translated_text[:70]}{'...' if len(translated_text) > 70 else ''}'"
        )

        return translated_text