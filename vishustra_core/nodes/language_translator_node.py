import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra processing node responsible for simulating text translation
    from an assumed source language to a specified target language.

    This node expects the input `data` to be a string representing the text
    to be translated. The `context` dictionary must contain a 'target_language'
    key, specifying the language code (e.g., 'es' for Spanish, 'fr' for French)
    for the translation.

    In a production environment, this node would integrate with an external
    translation service API (e.g., Google Cloud Translation, DeepL, Azure Translator).
    For this simulation, it appends a suffix indicating the translation.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of this processing node."""
        return "LanguageTranslator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by simulating its translation to the specified
        target language.

        Args:
            data (Any): The input data to be processed. Expected to be a string
                        containing the text to translate.
            context (Dict[str, Any]): A dictionary providing additional context.
                                      Must include 'target_language' (str)
                                      to specify the language for translation.

        Returns:
            Any: The simulated translated string.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If 'target_language' is missing from the `context`
                        or is not a non-empty string.
        """
        logger.debug(f"[{self.node_name}] Initiating translation process.")

        if not isinstance(data, str):
            error_message = (
                f"[{self.node_name}] Invalid input data type for translation. "
                f"Expected string, received {type(data).__name__}."
            )
            logger.error(error_message)
            raise TypeError(error_message)

        target_language = context.get("target_language")
        if not isinstance(target_language, str) or not target_language.strip():
            error_message = (
                f"[{self.node_name}] 'target_language' is missing or invalid in context. "
                "Expected a non-empty string, e.g., {'target_language': 'es'}."
            )
            logger.error(error_message)
            raise ValueError(error_message)
        
        target_language_cleaned = target_language.strip()

        # --- Simulation of translation ---
        # In a real-world scenario, this section would involve:
        # 1. API client initialization for a translation service.
        # 2. Making an API call with `data` and `target_language_cleaned`.
        # 3. Handling potential API errors (network issues, rate limits, invalid language codes).
        # 4. Parsing the response to extract the translated text.

        # For demonstration purposes, we append a suffix.
        translated_data = f"{data}_translated_to_{target_language_cleaned}"
        # --- End of simulation ---

        logger.info(
            f"[{self.node_name}] Successfully simulated translation to "
            f"'{target_language_cleaned}'. Original text length: {len(data)}."
        )
        logger.debug(f"[{self.node_name}] Original: '{data[:75]}...'")
        logger.debug(f"[{self.node_name}] Translated: '{translated_data[:75]}...'")

        return translated_data
