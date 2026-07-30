import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra processing node that simulates language translation of text data.

    This node expects the input 'data' to be a string and the 'context' dictionary
    to contain a 'target_language' key specifying the language to translate into.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "LanguageTranslatorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by simulating translation to a target language.

        Args:
            data: The input text data to be translated (expected to be a string).
            context: A dictionary containing processing context.
                     Must include 'target_language' (str) e.g., 'es', 'fr', 'de'.

        Returns:
            The simulated translated text data.

        Raises:
            TypeError: If 'data' is not a string.
            ValueError: If 'target_language' is missing or not a valid string in 'context'.
        """
        logger.debug(
            "Entering LanguageTranslatorNode.process for data of type '%s'.",
            type(data).__name__
        )

        if not isinstance(data, str):
            logger.error(
                "LanguageTranslatorNode received non-string data. Expected string, got %s.",
                type(data).__name__
            )
            raise TypeError(
                f"LanguageTranslatorNode expects string data for translation, "
                f"received {type(data).__name__}."
            )

        target_language = context.get("target_language")
        if not isinstance(target_language, str) or not target_language.strip():
            logger.error(
                "LanguageTranslatorNode requires 'target_language' (non-empty string) in context. Got: %s",
                repr(target_language)
            )
            raise ValueError(
                "Missing or invalid 'target_language' in context for LanguageTranslatorNode. "
                "Expected a non-empty string."
            )
        
        target_language = target_language.strip().upper() # Normalize target language for consistent tagging

        logger.info(
            "Initiating translation of text (length: %d) to '%s' via LanguageTranslatorNode.",
            len(data), target_language
        )

        # Simulate translation. In a real-world scenario, this would involve
        # an API call to a translation service (e.g., Google Translate, DeepL)
        # or invocation of an in-house ML model.
        # For this simulation, we append the target language tag.
        translated_data = f"{data} [{target_language}]"

        logger.debug(
            "Translation complete. Original (first 50 chars): '%s', Translated (first 50 chars): '%s'",
            data[:50], translated_data[:50]
        )
        return translated_data