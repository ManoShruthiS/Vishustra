import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra node that simulates language translation of text data.

    This node takes a string as input and translates it to a specified
    target language using a predefined, simulated translation map.
    It expects 'target_language' to be provided in the context dictionary.
    """

    # A simple, simulated translation map for demonstration purposes.
    # In a real-world scenario, this would interface with an external
    # translation service (e.g., Google Translate API, DeepL, etc.).
    _SIMULATED_TRANSLATION_MAP = {
        "Hello, world!": {
            "es": "¡Hola, mundo!",
            "fr": "Bonjour, le monde!"
        },
        "How are you?": {
            "es": "¿Cómo estás?",
            "fr": "Comment allez-vous?"
        },
        "Please translate this.": {
            "es": "Por favor, traduce esto.",
            "fr": "Veuillez traduire ceci."
        },
        "Thank you for your help.": {
            "es": "Gracias por tu ayuda.",
            "fr": "Merci pour votre aide."
        },
        "This is a test.": {
            "es": "Esto es una prueba.",
            "fr": "Ceci est un test."
        }
    }

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "LanguageTranslator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by simulating translation to a target language.

        Args:
            data: The text (string) to be translated.
            context: A dictionary containing operational parameters, expected to include:
                     - 'target_language' (str): The desired language code (e.g., 'es', 'fr').

        Returns:
            The translated string if a translation is found in the simulated map
            for the given input and target language. If no translation is found
            or the target language is not supported for the phrase, the original
            data is returned after logging a warning.

        Raises:
            ValueError: If 'data' is not a string or if 'target_language' is
                        missing or invalid in the context.
        """
        logger.debug(f"[{self.node_name}] Starting process for input data type: {type(data)}")

        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', but received '{type(data).__name__}'.")
            raise ValueError(
                f"[{self.node_name}] Input data must be a string for translation."
            )

        target_language = context.get("target_language")
        if not isinstance(target_language, str) or not target_language:
            logger.error(
                f"[{self.node_name}] Missing or invalid 'target_language' in context. "
                f"Expected a non-empty string, but received '{target_language}'.")
            raise ValueError(
                f"[{self.node_name}] Context must contain a valid 'target_language' string."
            )

        # Normalize target language for consistent lookup (e.g., 'es' vs 'ES')
        normalized_target_language = target_language.lower()

        # Attempt to find the translation in the simulated map
        if data in self._SIMULATED_TRANSLATION_MAP:
            translations_for_phrase = self._SIMULATED_TRANSLATION_MAP[data]
            if normalized_target_language in translations_for_phrase:
                translated_text = translations_for_phrase[normalized_target_language]
                logger.info(
                    f"[{self.node_name}] Successfully translated '{data}' to '{target_language}'.")
                return translated_text
            else:
                logger.warning(
                    f"[{self.node_name}] Target language '{target_language}' is not supported "
                    f"for phrase '{data}' in the simulated translation map. Returning original data."
                )
                return data
        else:
            logger.warning(
                f"[{self.node_name}] Phrase '{data}' not found in the simulated translation map. "
                f"Returning original data."
            )
            return data