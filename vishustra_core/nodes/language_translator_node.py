import logging
from typing import Any, Dict, Union

# Assuming vishustra_core exists and base_node is within it
# This import path needs to match the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra processing node that simulates language translation.

    This node takes text content and simulates its translation from a source
    language to a target language. It supports both string and dictionary
    inputs, attempting to translate the 'text' field if a dictionary is provided.

    Translation parameters can be set during initialization or overridden via
    the context dictionary at processing time.
    """

    def __init__(self, target_language: str = "en", source_language: str = "auto") -> None:
        """
        Initializes the LanguageTranslatorNode.

        Args:
            target_language (str): The default language code to translate into (e.g., 'en', 'fr', 'es').
            source_language (str): The default source language code, or 'auto' for automatic detection.
        """
        if not isinstance(target_language, str) or not target_language:
            raise ValueError("target_language must be a non-empty string.")
        if not isinstance(source_language, str) or not source_language:
            raise ValueError("source_language must be a non-empty string.")

        self._default_target_language = target_language.lower()
        self._default_source_language = source_language.lower()
        logger.debug(
            f"LanguageTranslatorNode initialized with default target_language='{self._default_target_language}' "
            f"and source_language='{self._default_source_language}'."
        )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "LanguageTranslator"

    def process(self, data: Union[str, Dict[str, Any]], context: Dict[str, Any]) -> Union[str, Dict[str, Any]]:
        """
        Processes the input data by simulating language translation.

        Args:
            data (Union[str, Dict[str, Any]]): The input data to be translated.
                                                If a string, it's treated as the text to translate.
                                                If a dictionary, it expects a 'text' key containing
                                                the string to translate.
            context (Dict[str, Any]): A dictionary containing additional runtime information.
                                      Can include 'target_language' and 'source_language' to
                                      override the node's defaults for this specific invocation.

        Returns:
            Union[str, Dict[str, Any]]: The translated text. If the input was a string, a string
                                        is returned. If the input was a dictionary, a dictionary
                                        is returned with 'translated_text', 'source_language_detected',
                                        and 'target_language_used' fields added or updated.

        Raises:
            ValueError: If the input data is not a string or a dictionary with a 'text' key.
            TypeError: If the 'text' within a dictionary is not a string.
        """
        text_to_translate: str = ""
        is_dict_input: bool = False

        if isinstance(data, str):
            text_to_translate = data
        elif isinstance(data, dict):
            is_dict_input = True
            if "text" in data:
                if isinstance(data["text"], str):
                    text_to_translate = data["text"]
                else:
                    logger.error(f"LanguageTranslatorNode received a dictionary where 'text' key is not a string. Type: {type(data['text'])}")
                    raise TypeError("The 'text' field in the input dictionary must be a string.")
            else:
                logger.error("LanguageTranslatorNode received a dictionary input without a 'text' key.")
                raise ValueError("Input dictionary must contain a 'text' key for translation.")
        else:
            logger.error(f"LanguageTranslatorNode received invalid input data type: {type(data)}. Expected str or dict.")
            raise ValueError("Invalid input data for LanguageTranslatorNode. Expected string or dictionary with 'text' key.")

        if not text_to_translate.strip():
            logger.warning("LanguageTranslatorNode received empty or whitespace-only text for translation. Returning as-is.")
            if is_dict_input:
                result = data.copy()
                result["translated_text"] = ""
                result["source_language_detected"] = context.get("source_language", self._default_source_language)
                result["target_language_used"] = context.get("target_language", self._default_target_language)
                return result
            return ""


        # Determine languages to use for this processing step
        current_target_lang = context.get("target_language", self._default_target_language).lower()
        current_source_lang = context.get("source_language", self._default_source_language).lower()

        logger.info(
            f"Translating text (first 60 chars: '{text_to_translate[:60].replace('\'', '\"')}...') "
            f"from '{current_source_lang}' to '{current_target_lang}'."
        )

        # Simulate translation logic. In a real scenario, this would involve
        # calling an external translation service or a sophisticated local model.
        # For demonstration, we simply prepend a tag.
        translated_text = f"[{current_source_lang.upper()}->{current_target_lang.upper()}] {text_to_translate}"

        if is_dict_input:
            # If the input was a dictionary, return an updated dictionary
            result = data.copy()
            result["translated_text"] = translated_text
            # Simulate language detection and usage for context
            result["source_language_detected"] = current_source_lang
            result["target_language_used"] = current_target_lang
            logger.debug("LanguageTranslatorNode successfully processed dictionary input.")
            return result
        else:
            # If the input was a string, return just the translated string
            logger.debug("LanguageTranslatorNode successfully processed string input.")
            return translated_text

# Basic logging configuration for local testing if run directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Mocking the BaseNode for local execution, normally it would be imported from a package
    class MockBaseNode(ABC):
        @abstractmethod
        def process(self, data: Any, context: Dict[str, Any]) -> Any: pass
        @property
        @abstractmethod
        def node_name(self) -> str: pass

    # Override the import for local testing purposes to run the example below
    # In a real project, this would not be needed as vishustra_core would be installed
    try:
        from vishustra_core.nodes.base_node import BaseNode # type: ignore
    except ImportError:
        # If running this file directly for testing, use a mock BaseNode
        # This allows the example below to run without a full project setup
        logger.warning("Could not import BaseNode from vishustra_core.nodes.base_node. Using a mock BaseNode for local testing.")
        BaseNode = MockBaseNode # type: ignore


    # --- Example Usage ---
    translator_fr = LanguageTranslatorNode(target_language="fr")
    translator_es_auto = LanguageTranslatorNode(target_language="es", source_language="en")

    # Test 1: String input
    text1 = "Hello, how are you?"
    translated_text1 = translator_fr.process(text1, {})
    logger.info(f"Original: '{text1}'\nTranslated: '{translated_text1}'")
    assert translated_text1 == "[AUTO->FR] Hello, how are you?"

    # Test 2: Dictionary input
    data2 = {"id": 1, "text": "This is a test sentence.", "priority": "high"}
    translated_data2 = translator_es_auto.process(data2, {})
    logger.info(f"Original: '{data2}'\nTranslated: '{translated_data2}'")
    assert translated_data2["translated_text"] == "[EN->ES] This is a test sentence."
    assert translated_data2["target_language_used"] == "es"
    assert translated_data2["id"] == 1 # Ensure other data is preserved

    # Test 3: Override target_language via context
    text3 = "The quick brown fox jumps over the lazy dog."
    translated_text3 = translator_fr.process(text3, {"target_language": "de"})
    logger.info(f"Original: '{text3}' (override target to DE)\nTranslated: '{translated_text3}'")
    assert translated_text3 == "[AUTO->DE] The quick brown fox jumps over the lazy dog."

    # Test 4: Empty string input
    text4 = "   "
    translated_text4 = translator_fr.process(text4, {})
    logger.info(f"Original: '{text4}'\nTranslated: '{translated_text4}'")
    assert translated_text4 == ""

    # Test 5: Dictionary with empty text
    data5 = {"text": " ", "meta": "info"}
    translated_data5 = translator_es_auto.process(data5, {})
    logger.info(f"Original: '{data5}'\nTranslated: '{translated_data5}'")
    assert translated_data5["translated_text"] == ""

    # Test 6: Invalid input type (list)
    try:
        translator_fr.process(["list item"], {})
    except ValueError as e:
        logger.info(f"Caught expected error for invalid input type: {e}")
        assert "Invalid input data" in str(e)

    # Test 7: Dictionary without 'text' key
    try:
        translator_fr.process({"id": 1, "value": "some data"}, {})
    except ValueError as e:
        logger.info(f"Caught expected error for dict without 'text' key: {e}")
        assert "Input dictionary must contain a 'text' key" in str(e)

    # Test 8: Dictionary with non-string 'text' key
    try:
        translator_fr.process({"text": 12345}, {})
    except TypeError as e:
        logger.info(f"Caught expected error for non-string 'text' key: {e}")
        assert "The 'text' field in the input dictionary must be a string." in str(e)

    logger.info("All LanguageTranslatorNode tests passed successfully!")