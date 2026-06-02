import logging
from typing import Any, Dict, Optional

# Assuming BaseNode is available at this path as per project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra processing node designed to translate text from an input language
    to a specified target language.

    This node simulates a translation service, expecting string data as input.
    It can optionally utilize a 'source_language' hint from the context dictionary
    for logging purposes or more advanced real-world translation API calls.
    """

    def __init__(self, target_language: str, translation_service_config: Optional[Dict[str, Any]] = None):
        """
        Initializes the LanguageTranslatorNode with a mandatory target language
        and an optional configuration for a simulated translation service.

        Args:
            target_language: The ISO 639-1 code for the language to translate into
                             (e.g., 'es' for Spanish, 'fr' for French).
                             This is the fixed language this node instance will translate towards.
            translation_service_config: A dictionary representing configuration for a
                                        translation service. For simulation purposes, this
                                        can contain a 'simulated_phrases' map, structured as:
                                        {'original_phrase': {'target_lang_code': 'translated_phrase', ...}, ...}
                                        Example:
                                        {'Hello': {'es': 'Hola', 'fr': 'Bonjour'},
                                         'How are you?': {'es': '¿Cómo estás?'}}
        Raises:
            ValueError: If 'target_language' is not a valid non-empty string.
        """
        if not isinstance(target_language, str) or not target_language.strip():
            logger.error(
                "LanguageTranslatorNode requires a valid non-empty string for 'target_language'. "
                f"Received: '{target_language}'."
            )
            raise ValueError("target_language must be a non-empty string.")

        self._target_language = target_language.strip()
        self._translation_service_config = translation_service_config or {}
        
        # Simulate a translation "model" or "engine" based on config.
        # This structure allows an original phrase to have multiple target language translations,
        # and the node selects based on its configured _target_language.
        self._simulated_translations: Dict[str, Dict[str, str]] = self._translation_service_config.get(
            'simulated_phrases', {}
        )
        
        logger.info(f"LanguageTranslatorNode initialized to translate to '{self._target_language}'.")
        if self._simulated_translations:
            logger.debug(f"Loaded {len(self._simulated_translations)} simulated translation phrase entries.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "LanguageTranslator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by translating it to the configured target language.
        If translation is not possible via the simulated service, the original data is returned.

        Args:
            data: The string text to be translated.
            context: A dictionary containing contextual information.
                     Can optionally contain 'source_language' (str) as a hint
                     for the translation process (e.g., 'en', 'de').

        Returns:
            The translated text as a string. If the input data is not a string,
            or if translation fails within the simulated service, the original
            text is returned after appropriate logging.

        Raises:
            TypeError: If the input 'data' is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"Invalid input data type for LanguageTranslatorNode. Expected str, "
                f"got {type(data).__name__}. This node only processes string data."
            )
            raise TypeError("LanguageTranslatorNode expects string data for translation.")

        input_text = data
        # 'source_language' from context can be used by real translation APIs
        # but for our simple simulated map, it's primarily for logging.
        source_language_hint = context.get('source_language') 
        
        translated_text = input_text # Initialize with original text as default fallback

        # In a real-world scenario, this block would invoke an external translation API
        # (e.g., Google Translate, DeepL, Azure Translator).
        # For this simulation, we use the pre-configured '_simulated_translations' map.
        
        if input_text in self._simulated_translations:
            phrase_translations = self._simulated_translations[input_text]
            
            # Check if there's a specific translation for *this node's* target_language
            if isinstance(phrase_translations, dict) and self._target_language in phrase_translations:
                translated_text = phrase_translations[self._target_language]
                logger.debug(
                    f"Successfully translated '{input_text[:50]}...' to '{translated_text[:50]}...' "
                    f"(target: {self._target_language}) using simulated phrases."
                )
            else:
                logger.warning(
                    f"Simulated phrase found for '{input_text[:50]}...', but no specific translation "
                    f"for target language '{self._target_language}'. Returning original text."
                )
        else:
            logger.debug(
                f"No exact simulated translation phrase found for '{input_text[:50]}...'. "
                f"Returning original text."
            )

        if translated_text == input_text:
            logger.warning(
                f"Translation attempt for '{input_text[:100]}...' to '{self._target_language}' "
                f"failed or no match found in simulated phrases. Original text was returned."
            )
        else:
            logger.info(
                f"Translated text from (assumed) '{source_language_hint or 'unknown'}' "
                f"to '{self._target_language}'. "
                f"Original: '{input_text[:50]}...', Translated: '{translated_text[:50]}...'"
            )

        return translated_text