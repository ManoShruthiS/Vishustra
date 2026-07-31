import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra node designed to simulate language translation for text content.
    This node expects the input `data` to be a string and requires a
    'target_language' specified in the `context` dictionary.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "LanguageTranslatorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by simulating translation to a specified target language.

        The translation mechanism is simulated for demonstration purposes. In a
        production environment, this would involve integration with a dedicated
        translation service (e.g., Google Translate API, DeepL, etc.).

        Args:
            data: The text content (expected to be a string) to be translated.
            context: A dictionary containing operational parameters for translation.
                     It *must* include 'target_language' (str) indicating the
                     desired language for the translated output.
                     It can optionally include 'source_language' (str) for the
                     original text's language; defaults to 'auto' if not provided.

        Returns:
            The simulated translated text as a string.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If 'target_language' is missing from the `context` or is not
                        a valid non-empty string.
            Exception: Propagates any unexpected errors that occur during the
                       simulated translation process.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"but received '{type(data).__name__}'."
            )
            raise TypeError(
                f"Input data for {self.node_name} must be a string, "
                f"but received {type(data).__name__}"
            )

        target_language = context.get("target_language")
        if not isinstance(target_language, str) or not target_language:
            logger.error(
                f"[{self.node_name}] Missing or invalid 'target_language' in context. "
                f"Expected a non-empty string, but got '{target_language}'."
            )
            raise ValueError(
                f"Context for {self.node_name} must contain a valid "
                f"'target_language' string."
            )

        source_language = context.get("source_language", "auto")

        logger.info(
            f"[{self.node_name}] Attempting to translate text from "
            f"'{source_language}' to '{target_language}'."
        )

        try:
            # Simulate translation: prepend a prefix to indicate translation
            # In a real scenario, this would involve an API call or a translation library.
            translated_text = f"[TRANSLATED to {target_language.upper()}] {data}"
            logger.info(
                f"[{self.node_name}] Successfully simulated translation to "
                f"'{target_language}'. Original length: {len(data)}, "
                f"Translated length: {len(translated_text)}."
            )
            return translated_text
        except Exception as e:
            logger.exception(
                f"[{self.node_name}] An unexpected error occurred during "
                f"simulated translation: {e}"
            )
            # Re-raise the exception to propagate the failure up the Vishustra orchestration
            raise
