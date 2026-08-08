import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra processing node that simulates language translation of text data.

    It expects a string as input data and a 'target_language' key in the context.
    A 'source_language' can also be specified in the context, defaulting to 'en'.
    """

    def __init__(self):
        """
        Initializes the LanguageTranslatorNode with a rudimentary internal
        translation map for demonstration purposes.
        """
        # In a production system, this would integrate with an actual translation service API.
        # This map serves to simulate that functionality.
        self._translation_map = {
            "en": {
                "hello": {"es": "hola", "fr": "bonjour", "de": "hallo", "it": "ciao"},
                "world": {"es": "mundo", "fr": "monde", "de": "welt", "it": "mondo"},
                "goodbye": {"es": "adiós", "fr": "au revoir", "de": "auf wiedersehen", "it": "arrivederci"},
                "thank you": {"es": "gracias", "fr": "merci", "de": "danke", "it": "grazie"},
            }
            # More languages and words would be added here in a real scenario
        }
        logger.debug(f"'{self.node_name}' node initialized with rudimentary translation map.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "LanguageTranslator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by simulating translation to a target language.

        Args:
            data: The input text data to be translated (expected to be a string).
            context: A dictionary containing runtime information, including:
                     - 'target_language' (str): The language to translate to (e.g., 'es', 'fr').
                     - 'source_language' (str, optional): The original language of the data.
                                                          Defaults to 'en' for simulation.

        Returns:
            str: The translated text, or the original text with a suffix if
                 no translation is found in the internal map.

        Raises:
            ValueError: If 'data' is not a string or 'target_language' is missing from context.
            RuntimeError: If an unexpected error occurs during the simulated translation process.
        """
        logger.info(f"'{self.node_name}' node starting process for data (truncated): '{str(data)[:100]}'")

        if not isinstance(data, str):
            error_msg = (f"'{self.node_name}' received non-string data. "
                         f"Expected str, got {type(data).__name__}.")
            logger.error(error_msg)
            raise ValueError(error_msg)

        target_language = context.get("target_language")
        if not target_language:
            error_msg = (f"'{self.node_name}' requires 'target_language' in context "
                         f"for translation. Context received: {list(context.keys())}.")
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Default to English as source for our simulation map
        source_language = context.get("source_language", "en")
        translated_text = data

        try:
            # Convert to lower case for simple map lookup in simulation
            normalized_data = data.lower()

            if (source_language in self._translation_map and
                    normalized_data in self._translation_map[source_language]):
                
                language_specific_translations = self._translation_map[source_language][normalized_data]

                if target_language in language_specific_translations:
                    translated_text = language_specific_translations[target_language]
                    logger.debug(f"Translated '{data}' from '{source_language}' to '{target_language}' as '{translated_text}' using map.")
                else:
                    logger.warning(f"No direct translation for '{data}' from '{source_language}' to "
                                   f"'{target_language}' found in map. Appending suffix.")
                    translated_text = f"{data}_translated_to_{target_language}"
            else:
                logger.debug(f"'{data}' not found in internal translation map for source language '{source_language}'. Appending suffix.")
                translated_text = f"{data}_translated_to_{target_language}"

        except Exception as e:
            logger.error(f"'{self.node_name}' encountered an unexpected error during simulated translation "
                         f"for data '{data}': {e}", exc_info=True)
            raise RuntimeError(f"Translation simulation failed for '{data}': {e}") from e

        logger.info(f"'{self.node_name}' successfully processed. Original: '{data}', Translated: '{translated_text}'.")
        return translated_text