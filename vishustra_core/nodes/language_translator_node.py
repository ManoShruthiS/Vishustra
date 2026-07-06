import logging
from typing import Any, Dict, Optional

# Assuming the project structure places BaseNode here
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra processing node that simulates language translation of text data.

    This node expects a string as input data and utilizes the 'target_language'
    key from the context dictionary to determine the desired translation.
    It provides a robust simulation, logging warnings for unsupported translations
    or missing context parameters, and raising errors for invalid input types.
    """

    # A simple internal map to simulate translations for common phrases.
    # In a real-world scenario, this would interface with a translation API.
    _SIMULATED_TRANSLATIONS = {
        "Hello": {"es": "Hola", "fr": "Bonjour", "de": "Hallo"},
        "Goodbye": {"es": "Adiós", "fr": "Au revoir", "de": "Auf Wiedersehen"},
        "Thank you": {"es": "Gracias", "fr": "Merci", "de": "Danke schön"},
        "Please": {"es": "Por favor", "fr": "S'il vous plaît", "de": "Bitte"},
        "Yes": {"es": "Sí", "fr": "Oui", "de": "Ja"},
        "No": {"es": "No", "fr": "Non", "de": "Nein"},
    }

    def __init__(self):
        """
        Initializes the LanguageTranslatorNode.
        """
        logger.debug(f"[{self.node_name}] Initializing node.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "LanguageTranslator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by attempting to translate it to a target language.

        The `data` is expected to be a string representing the text to be translated.
        The `context` dictionary *must* contain a 'target_language' key,
        whose value is a string (e.g., 'es' for Spanish, 'fr' for French, 'de' for German).

        Args:
            data: The string text content to be translated.
            context: A dictionary containing operational parameters.
                     Expected to contain 'target_language' (str).

        Returns:
            The translated string if a simulated translation is found for the
            given text and target language. Otherwise, the original data is
            returned, and a warning is logged.

        Raises:
            ValueError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"but received '{type(data).__name__}'. Unable to translate."
            )
            raise ValueError(f"Input data for {self.node_name} must be a string.")

        target_language: Optional[str] = context.get("target_language")

        if not target_language:
            logger.warning(
                f"[{self.node_name}] 'target_language' not found in context. "
                "Returning original data without translation."
            )
            return data

        # Normalize target language for lookup
        target_language = target_language.lower()
        original_text_stripped = data.strip()

        # Attempt to simulate translation
        translations_for_text = self._SIMULATED_TRANSLATIONS.get(original_text_stripped)

        if translations_for_text:
            translated_text = translations_for_text.get(target_language)
            if translated_text:
                logger.info(
                    f"[{self.node_name}] Successfully translated "
                    f"'{original_text_stripped}' to '{translated_text}' "
                    f"for target language '{target_language}'."
                )
                return translated_text
            else:
                logger.warning(
                    f"[{self.node_name}] No simulated translation available for "
                    f"'{original_text_stripped}' to target language '{target_language}'. "
                    "Returning original data."
                )
                return data
        else:
            logger.warning(
                f"[{self.node_name}] No simulated translation entry found for "
                f"'{original_text_stripped}'. Returning original data."
            )
            return data
