import logging
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

# A simple, mock translation dictionary for demonstration purposes.
# In a real scenario, this would interface with a robust translation service.
_MOCK_TRANSLATIONS: Dict[str, Dict[str, Dict[str, str]]] = {
    "en": {
        "hello": {"es": "hola", "fr": "bonjour", "de": "hallo", "it": "ciao"},
        "world": {"es": "mundo", "fr": "monde", "de": "welt", "it": "mondo"},
        "goodbye": {"es": "adiós", "fr": "au revoir", "de": "auf wiedersehen", "it": "arrivederci"},
        "thank you": {"es": "gracias", "fr": "merci", "de": "danke", "it": "grazie"},
        "how are you": {"es": "¿cómo estás?", "fr": "comment allez-vous?", "de": "wie geht es dir?", "it": "come stai?"},
    }
    # Expand with more source languages and phrases as needed for a more comprehensive mock
}


class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra node designed to translate text from one language to another.

    This node expects text data (string or within a dictionary) and a target
    language in the context. It simulates translation using an internal mock
    dictionary.

    Context parameters:
    - 'target_language' (str): The ISO 639-1 code of the language to translate into (e.g., 'es', 'fr').
    - 'source_language' (str, optional): The ISO 639-1 code of the source language. Defaults to 'en'.
    - 'fields_to_translate' (List[str], optional): If data is a dictionary, a list of keys
      whose string values should be translated. If omitted, and data is a dict,
      no translation occurs.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "LanguageTranslator"

    def _translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Simulates translation of a single text string.
        """
        text_lower = text.lower()
        if source_lang not in _MOCK_TRANSLATIONS:
            logger.warning(f"Unsupported source language '{source_lang}' for mock translation. Returning original text.")
            return text

        source_dict = _MOCK_TRANSLATIONS[source_lang]
        if text_lower in source_dict and target_lang in source_dict[text_lower]:
            translated_text = source_dict[text_lower][target_lang]
            logger.debug(f"Translated '{text}' from {source_lang} to {target_lang} as '{translated_text}'.")
            return translated_text
        else:
            logger.info(
                f"No mock translation found for '{text}' from {source_lang} to {target_lang}. "
                f"Returning original text with a prefix."
            )
            # For unmocked phrases, return with a marker to show it passed through
            return f"[Translated to {target_lang.upper()}]: {text}"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by translating it to the specified target language.

        Args:
            data (Any): The input data. Expected to be a string for direct translation,
                        or a dictionary where specific string values can be translated.
            context (Dict[str, Any]): A dictionary containing contextual information,
                                      including 'target_language' and optionally 'source_language'
                                      or 'fields_to_translate'.

        Returns:
            Any: The translated data. Returns original data if translation fails or
                 is not applicable.

        Raises:
            ValueError: If 'target_language' is missing from the context.
        """
        target_language = context.get("target_language")
        source_language = context.get("source_language", "en") # Default to English source

        if not target_language:
            logger.error("LanguageTranslatorNode requires 'target_language' in context for translation.")
            raise ValueError("Missing 'target_language' in context.")

        if target_language == source_language:
            logger.info(f"Target language '{target_language}' is the same as source language. Skipping translation.")
            return data

        if isinstance(data, str):
            logger.debug(f"Translating string data from {source_language} to {target_language}.")
            return self._translate_text(data, source_language, target_language)

        elif isinstance(data, dict):
            fields_to_translate: Union[List[str], None] = context.get("fields_to_translate")
            if not fields_to_translate:
                logger.debug("Data is a dictionary but 'fields_to_translate' not specified in context. "
                             "Returning dictionary unchanged.")
                return data

            translated_data = data.copy()
            for field in fields_to_translate:
                if field in translated_data and isinstance(translated_data[field], str):
                    logger.debug(f"Translating field '{field}' in dictionary data.")
                    translated_data[field] = self._translate_text(
                        translated_data[field], source_language, target_language
                    )
                else:
                    logger.warning(
                        f"Field '{field}' not found or not a string in dictionary data. Skipping translation for this field."
                    )
            return translated_data

        else:
            logger.warning(
                f"Unsupported data type for translation: {type(data)}. "
                f"Expected str or dict. Returning original data."
            )
            return data