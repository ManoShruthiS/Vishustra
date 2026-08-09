import logging
from typing import Any, Dict

# Assuming vishustra_core.nodes.base_node exists in the project structure
from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for this module
logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra processing node that simulates language translation of text data.

    This node expects string data as input and requires a 'target_language'
    key in the processing context. It can optionally use 'source_language'
    from the context for informational logging.

    The translation operation is simulated by appending a language tag to the
    input text. In a production environment, this would integrate with a
    dedicated translation service or model.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "LanguageTranslator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by simulating translation to a specified target language.

        Args:
            data (Any): The input data, expected to be a string containing the text to translate.
            context (Dict[str, Any]): A dictionary containing contextual information
                                      required for processing. Must include:
                                      - 'target_language' (str): The language code to translate into (e.g., 'es', 'fr').
                                      Can optionally include:
                                      - 'source_language' (str): The language code of the input text (e.g., 'en').
                                                                 If not provided, 'auto-detected' is assumed for logging.

        Returns:
            Any: The simulated translated text as a string.

        Raises:
            ValueError: If the input 'data' is not a string, or if 'target_language'
                        is missing from the context.
            TypeError: If 'target_language' in the context is not a string.
        """
        # Validate input data type
        if not isinstance(data, str):
            error_message = (
                f"Node '{self.node_name}' received invalid input data type. "
                f"Expected 'str', but got '{type(data).__name__}'."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        # Retrieve and validate target_language from context
        target_language = context.get("target_language")
        if not target_language:
            error_message = (
                f"Node '{self.node_name}' requires 'target_language' in context "
                "for translation. None was provided."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        if not isinstance(target_language, str):
            error_message = (
                f"Node '{self.node_name}' received invalid type for 'target_language' "
                f"in context. Expected 'str', but got '{type(target_language).__name__}'."
            )
            logger.error(error_message)
            raise TypeError(error_message)

        # Retrieve source_language from context, with a default for logging if not provided
        source_language = context.get("source_language", "auto-detected")
        if not isinstance(source_language, str):
            logger.warning(
                f"Node '{self.node_name}' received invalid type for 'source_language' "
                f"in context. Expected 'str', but got '{type(source_language).__name__}'. "
                "Proceeding with 'auto-detected' for logging."
            )
            source_language = "auto-detected" # Ensure source_language is a string for f-string

        logger.info(
            f"Node '{self.node_name}' initiated translation from '{source_language}' "
            f"to '{target_language}'. Original text length: {len(data)} characters."
        )

        # Simulate the translation process.
        # In a real-world scenario, this would involve an API call to a translation service
        # or an interaction with an integrated ML model.
        simulated_translated_text = f"{data} [translated_to_{target_language}]"

        logger.info(
            f"Node '{self.node_name}' successfully simulated translation to "
            f"'{target_language}'. Result length: {len(simulated_translated_text)} characters."
        )
        return simulated_translated_text