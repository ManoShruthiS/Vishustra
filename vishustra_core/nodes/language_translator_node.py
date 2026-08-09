import logging
from typing import Any, Dict, Optional

# Assuming BaseNode is available at this path as per project instructions
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra node that simulates language translation of text data.

    This node is designed to take a string as input `data` and translate
    it based on language specifications provided in the `context`.
    It expects 'target_language' to be present in the `context` dictionary.
    Optionally, a 'source_language' can also be provided; if omitted, it
    defaults to 'en' for simulation purposes.
    """

    # A simplified, in-memory translation map for demonstration.
    # In a real-world scenario, this would involve calling an external
    # LLM API or a dedicated translation service.
    _TRANSLATION_MAP = {
        "en": {
            "hello": {"es": "hola", "fr": "bonjour", "de": "hallo", "it": "ciao"},
            "goodbye": {"es": "adiós", "fr": "au revoir", "de": "auf wiedersehen", "it": "arrivederci"},
            "thank you": {"es": "gracias", "fr": "merci", "de": "danke schön", "it": "grazie"},
            "please": {"es": "por favor", "fr": "s'il vous plaît", "de": "bitte", "it": "per favore"},
            "how are you": {"es": "¿cómo estás?", "fr": "comment allez-vous?", "de": "wie geht es dir?", "it": "come stai?"},
            "data processing": {"es": "procesamiento de datos", "fr": "traitement des données", "de": "datenverarbeitung", "it": "elaborazione dati"},
            "orchestration framework": {"es": "marco de orquestación", "fr": "cadre d'orchestration", "de": "orchestrierungs-framework", "it": "framework di orchestrazione"},
        }
        # Additional source languages and their translations could be added here
        # to expand the simulation capabilities.
    }

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "LanguageTranslatorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by attempting to translate it to a specified
        target language.

        Args:
            data: The text string to be translated. This method expects a string.
            context: A dictionary containing parameters for translation.
                     Must include 'target_language' (str).
                     Can optionally include 'source_language' (str, defaults to 'en').

        Returns:
            The translated text string if a translation is found and successful.
            If translation fails (e.g., no mapping found, or service error in
            a real implementation), the original data is returned.

        Raises:
            ValueError: If 'data' is not a string, or if 'target_language' is
                        missing or invalid in the context.
        """
        if not isinstance(data, str):
            logger.error(
                "[%s] Invalid input data type. Expected 'str', got '%s'. Data: %s",
                self.node_name,
                type(data).__name__,
                data,
            )
            raise ValueError(
                f"[{self.node_name}] Input 'data' must be a string for translation."
            )

        target_language_raw: Optional[str] = context.get("target_language")
        source_language_raw: Optional[str] = context.get("source_language", "en") # Default to English for simulation

        if not target_language_raw or not isinstance(target_language_raw, str):
            logger.error(
                "[%s] 'target_language' is missing or not a string in context: %s",
                self.node_name,
                context,
            )
            raise ValueError(
                f"[{self.node_name}] 'target_language' must be a non-empty string in the context."
            )
        
        # Normalize language codes to lowercase for consistent dictionary lookup
        target_language: str = target_language_raw.lower()
        source_language: str = source_language_raw.lower()

        logger.debug(
            "[%s] Attempting to translate text from '%s' to '%s': '%s'",
            self.node_name,
            source_language,
            target_language,
            data
        )
        
        # Perform the simulated translation
        # Accessing the map requires converting input data to lowercase for consistency
        # with the keys in _TRANSLATION_MAP.
        translated_text: Optional[str] = self._TRANSLATION_MAP \
            .get(source_language, {}) \
            .get(data.lower(), {}) \
            .get(target_language)

        if translated_text:
            logger.info(
                "[%s] Successfully translated '%s' (from '%s') to '%s': '%s'",
                self.node_name,
                data,
                source_language,
                target_language,
                translated_text
            )
            return translated_text
        else:
            logger.warning(
                "[%s] No direct translation found for '%s' from '%s' to '%s' in internal map. Returning original data.",
                self.node_name,
                data,
                source_language,
                target_language
            )
            # In a production system, a real LLM call would be attempted here,
            # or a specific 'TranslationFailedException' might be raised if fallback
            # to original data is not desired.
            return data

