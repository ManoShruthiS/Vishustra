import logging
from typing import Any, Dict

# Assuming vishustra_core.nodes.base_node is available in the Python path
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra processing node designed to simulate language translation of text data.

    This node extracts text from the input `data` (which can be a string or a dictionary
    with a 'text' key) and simulates its translation into a `target_language`
    specified in the `context`. It provides robust input validation and error handling
    to ensure reliable operation within an orchestration flow.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "LanguageTranslatorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Simulates the translation of input text based on the `target_language`
        provided in the context.

        The node expects the input `data` to be either:
        1. A string: The text directly to be translated.
        2. A dictionary: Must contain a 'text' key whose value is the string to translate.

        The `context` dictionary must contain a 'target_language' key, which is
        a string representing the desired language code (e.g., 'es' for Spanish,
        'fr' for French, 'de' for German).

        Args:
            data: The input payload containing the text to be translated.
                  Expected types: `str` or `Dict[str, str]` (with a 'text' key).
            context: A dictionary containing operational parameters for the node,
                     most notably `{'target_language': 'en'}`.

        Returns:
            A string representing the simulated translated text. The simulation
            appends a tag indicating the target language.

        Raises:
            TypeError: If the `data` is not a supported type (str or dict),
                       or if a dictionary `data` does not contain a valid 'text' key.
            ValueError: If `target_language` is missing, empty, or not a string
                        in the `context`.
        """
        text_to_translate: str = ""

        # 1. Validate and extract text from input data
        if isinstance(data, str):
            text_to_translate = data
            logger.debug("Extracted text from direct string input.")
        elif isinstance(data, dict):
            if 'text' in data and isinstance(data['text'], str):
                text_to_translate = data['text']
                logger.debug("Extracted text from dictionary 'text' key.")
            else:
                error_msg = f"Input data dictionary missing 'text' key or 'text' value is not a string. Data: {data}"
                logger.error(error_msg)
                raise TypeError(error_msg)
        else:
            error_msg = f"Unsupported input data type: {type(data)}. Expected str or dict."
            logger.error(error_msg)
            raise TypeError(error_msg)

        # 2. Validate and extract target language from context
        target_language: str
        if 'target_language' in context and \
           isinstance(context['target_language'], str) and \
           context['target_language'].strip():
            target_language = context['target_language'].strip().lower()
            logger.debug(f"Target language '{target_language}' successfully extracted from context.")
        else:
            error_msg = f"Context must contain a valid 'target_language' (non-empty string). Context: {context}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 3. Simulate translation process
        # In a production system, this would involve an API call to an actual
        # Neural Machine Translation (NMT) service (e.g., DeepL, Google Cloud Translate).
        # For this node, we simulate by appending a language tag.
        simulated_translation = f"{text_to_translate} (translated to {target_language.upper()})"
        logger.info(
            f"Simulated translation of text (first 50 chars: '{text_to_translate[:50]}...') "
            f"to '{target_language}'. Result (first 50 chars): '{simulated_translation[:50]}...'"
        )

        return simulated_translation