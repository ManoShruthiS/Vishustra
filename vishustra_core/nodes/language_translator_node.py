import logging
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra processing node that simulates language translation.

    This node takes an input string and "translates" it to a specified target language.
    Translation is simulated using a simple internal mapping for demonstration purposes.
    In a production environment, this would integrate with an actual translation service
    or an internal model.
    """

    def __init__(self, default_target_language: str = "en",
                 mock_translations: Optional[Dict[str, Dict[str, str]]] = None):
        """
        Initializes the LanguageTranslatorNode with a default target language and
        an optional set of mock translations.

        Args:
            default_target_language: The language code (e.g., "en", "es", "fr") to
                                     translate to if 'target_language' is not provided
                                     in the `context` for a `process` call.
            mock_translations: A dictionary defining mock translations.
                               The structure should be:
                               `{"source_text": {"target_lang_code": "translated_text"}}`.
                               If None, a default set of mock translations is used.
        """
        self._default_target_language = default_target_language
        self._mock_translations = mock_translations if mock_translations is not None else {
            "Hello": {"es": "Hola", "fr": "Bonjour", "de": "Hallo"},
            "World": {"es": "Mundo", "fr": "Monde", "de": "Welt"},
            "Thank you": {"es": "Gracias", "fr": "Merci", "de": "Danke schön"},
            "Good morning": {"es": "Buenos días", "fr": "Bonjour", "de": "Guten Morgen"},
            "Please": {"es": "Por favor", "fr": "S'il vous plaît", "de": "Bitte"},
            "Yes": {"es": "Sí", "fr": "Oui", "de": "Ja"},
            "No": {"es": "No", "fr": "Non", "de": "Nein"},
            "How are you?": {"es": "¿Cómo estás?", "fr": "Comment allez-vous?", "de": "Wie geht es Ihnen?"},
            "Vishustra is great": {"es": "Vishustra es genial", "fr": "Vishustra est super", "de": "Vishustra ist großartig"},
        }
        logger.debug(f"LanguageTranslatorNode initialized with default_target_language: "
                     f"{self._default_target_language} and {len(self._mock_translations)} mock entries.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "LanguageTranslator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Translates the input data (expected to be a string) to a specified target language.

        The target language is determined in the following order:
        1. From `context["target_language"]` if present.
        2. From the `default_target_language` provided during node initialization.

        If the `data` is not a string, a `ValueError` is raised.
        If no target language can be determined, a `ValueError` is raised.
        If a direct mock translation is not available for the given text and target language,
        the original text is returned, and a warning is logged.

        Args:
            data: The text string to be translated.
            context: A dictionary that may contain 'target_language' (str),
                     e.g., `{"target_language": "es"}`.

        Returns:
            The translated text string, or the original text if no mock translation
            is found for the specified target language.

        Raises:
            ValueError: If 'data' is not a string, or if no target language is specified
                        or configured.
        """
        if not isinstance(data, str):
            logger.error(f"Invalid input data type for LanguageTranslatorNode. Expected str, got {type(data)}.")
            raise ValueError(
                f"LanguageTranslatorNode expects string input for translation, "
                f"but received type: {type(data).__name__}."
            )

        text_to_translate = data.strip()
        target_language = context.get("target_language", self._default_target_language)

        if not target_language:
            logger.error("No target language specified in context or node configuration for translation.")
            raise ValueError("Target language must be specified either in context or during node initialization.")

        logger.info(f"Attempting to translate text to '{target_language}': '{text_to_translate[:100]}...'")

        # Simulate translation using the mock data
        if text_to_translate in self._mock_translations:
            translations_for_text = self._mock_translations[text_to_translate]
            if target_language in translations_for_text:
                translated_text = translations_for_text[target_language]
                logger.debug(f"Successfully translated '{text_to_translate[:50]}...' to "
                             f"'{translated_text[:50]}...' for language '{target_language}'.")
                return translated_text
            else:
                logger.warning(
                    f"No mock translation available for text '{text_to_translate[:50]}...' "
                    f"to target language '{target_language}'. Returning original text."
                )
                # In a real system, an actual translation service might still attempt a translation
                # or indicate an unsupported language. Here, we simulate by returning the original.
                return text_to_translate
        else:
            logger.info(
                f"No exact mock translation found for '{text_to_translate[:50]}...' "
                f"in internal dictionary. Returning original text."
            )
            # For texts not present in our mock map, we return the original text
            # to simulate a translator that might not have every phrase pre-translated
            # or an API call that resulted in no change.
            return text_to_translate