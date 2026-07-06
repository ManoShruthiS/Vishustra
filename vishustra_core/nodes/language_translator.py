import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra processing node designed to simulate language translation of text data.
    This node expects a string as input data and a target language in the context
    to produce a translated output.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "LanguageTranslator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by simulating translation to a specified target language.

        The `data` input is expected to be a string of text that needs translation.
        The `context` dictionary must contain a 'target_language' key, specifying
        the language to which the text should be translated (e.g., "es", "fr", "de").

        Args:
            data: The input text data to be translated, expected as a string.
            context: A dictionary containing runtime parameters, expected to include:
                     - 'target_language' (str): The ISO 639-1 code or name of the
                                                target language for translation.

        Returns:
            A string representing the "translated" text.

        Raises:
            ValueError: If 'data' is not a string, or if 'target_language' is
                        missing from the context or is not a valid non-empty string.
        """
        logger.debug(f"[{self.node_name}] Attempting to process data of type: {type(data)}")

        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected a string, "
                f"but received {type(data)}. Cannot translate non-string data."
            )
            raise ValueError(f"Input data for '{self.node_name}' must be a string.")

        target_language = context.get("target_language")

        if not isinstance(target_language, str) or not target_language.strip():
            logger.error(
                f"[{self.node_name}] Missing or invalid 'target_language' in context. "
                "Expected a non-empty string for the target language."
            )
            raise ValueError(
                f"Context for '{self.node_name}' must contain a non-empty string "
                f"for 'target_language'."
            )

        # Simulate the translation process. In a production environment, this
        # would typically involve an API call to an external translation service
        # (e.g., Google Cloud Translate, DeepL, AWS Translate, etc.).
        # For this simulation, we simply append an indicator of the target language.
        translated_text = f"{data} (translated to {target_language.strip()})"

        logger.info(
            f"[{self.node_name}] Successfully simulated translation to "
            f"'{target_language.strip()}'. Original text snippet: "
            f"'{data[:50]}...'. Translated text snippet: '{translated_text[:50]}...'"
        )

        return translated_text