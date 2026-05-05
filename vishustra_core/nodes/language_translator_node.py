import logging
from typing import Any, Dict, Optional, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A processing node designed to simulate language translation of textual data.

    This node accepts text data, either directly as a string or embedded within
    a dictionary, and "translates" it into a specified target language.
    The translation is simulated for demonstration purposes, prepending a
    language tag to the original text.

    Configuration can be provided via the 'context' dictionary:
    - 'target_language': The ISO 639-1 code for the target language (e.g., 'es', 'fr', 'de').
                         Defaults to 'en' if not provided.
    - 'config': A nested dictionary for node-specific settings:
        - 'input_key': If 'data' is a dictionary, this specifies the key whose
                       value should be translated. Defaults to 'text'.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name for this processing node.
        """
        return "Language Translator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Simulates the translation of input text based on the target language
        specified in the context.

        The node expects the input `data` to be either:
        1. A direct string which will be treated as the text to translate.
        2. A dictionary containing the text under a key specified by
           `context['config']['input_key']`, defaulting to 'text'.

        The target language is read from `context['target_language']`,
        defaulting to 'en' if not provided.

        Args:
            data (Any): The input data to be processed. Expected to be a string
                        or a dictionary containing a string.
            context (Dict[str, Any]): A dictionary containing runtime context,
                                     including configuration settings like
                                     'target_language' and 'config' for 'input_key'.

        Returns:
            Any: The "translated" data. If the input was a string, a string is returned.
                 If the input was a dictionary, the dictionary with the specified
                 field translated is returned. Returns original data on error or
                 if no translatable content is found.
        """
        target_language = context.get('target_language', 'en').lower()
        if not target_language:
            logger.warning(
                f"[{self.node_name}] No 'target_language' specified in context. Defaulting to 'en'."
            )
            target_language = 'en'

        logger.debug(
            f"[{self.node_name}] Attempting to translate data to language: '{target_language}'."
        )

        translated_text: Optional[str] = None
        # Safely retrieve input_key from nested 'config' dict within context
        input_key = context.get('config', {}).get('input_key', 'text')

        try:
            if isinstance(data, str):
                text_to_translate = data
                logger.debug(f"[{self.node_name}] Input data is a direct string for translation.")
                translated_text = self._simulate_translation(text_to_translate, target_language)
                logger.info(
                    f"[{self.node_name}] Successfully 'translated' string data to '{target_language.upper()}'."
                )
                return translated_text
            elif isinstance(data, dict):
                text_to_translate = data.get(input_key)
                if text_to_translate is None:
                    logger.warning(
                        f"[{self.node_name}] Input dictionary does not contain '{input_key}' key for translation. "
                        "Returning original data without modification."
                    )
                    return data
                if not isinstance(text_to_translate, str):
                    logger.warning(
                        f"[{self.node_name}] Value for key '{input_key}' in dictionary is not a string "
                        f"(found: {type(text_to_translate).__name__}). "
                        "Returning original data without modification."
                    )
                    return data

                logger.debug(
                    f"[{self.node_name}] Input data is a dictionary, translating key '{input_key}'."
                )
                translated_text = self._simulate_translation(text_to_translate, target_language)
                
                # Create a shallow copy to avoid modifying the original input dictionary directly
                output_data = data.copy()
                output_data[input_key] = translated_text
                logger.info(
                    f"[{self.node_name}] Successfully 'translated' dictionary field '{input_key}' to '{target_language.upper()}'."
                )
                return output_data
            else:
                logger.error(
                    f"[{self.node_name}] Unsupported data type received: '{type(data).__name__}'. "
                    "Expected string or dictionary. Returning original data."
                )
                return data
        except Exception as e:
            # Catch any unexpected errors during processing
            logger.exception(
                f"[{self.node_name}] An unexpected error occurred during processing: {e}. "
                "Returning original data to prevent pipeline disruption."
            )
            return data

    def _simulate_translation(self, text: str, target_lang: str) -> str:
        """
        A private helper method to simulate the translation process.
        In a real-world scenario, this method would integrate with an external
        translation service (e.g., Google Translate API, DeepL, or an internal LLM).
        For this simulation, it prepends a tag indicating the target language.

        Args:
            text (str): The text string to be "translated".
            target_lang (str): The ISO 639-1 code of the target language.

        Returns:
            str: The simulated translated text.
        """
        # A very basic simulation: prepend a tag indicating the target language
        return f"[Translated to {target_lang.upper()}]: {text}"