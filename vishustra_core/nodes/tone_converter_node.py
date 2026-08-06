import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverterNode(BaseNode):
    """
    A Vishustra node designed to convert the tone of an input text.
    It accepts a string of text and a 'target_tone' specified within the context
    dictionary, then simulates a transformation to produce text in the desired tone.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, applying a tone conversion based on the
        'target_tone' specified in the context dictionary.

        Args:
            data (Any): The input data to be processed. Expected to be a string
                        representing the text whose tone needs to be converted.
            context (Dict[str, Any]): A dictionary containing parameters and
                                       additional information for processing.
                                       Must include 'target_tone' (str) indicating
                                       the desired output tone.

        Returns:
            Any: The tone-converted string.

        Raises:
            TypeError: If the input 'data' is not a string, or if 'target_tone'
                       in context is not a string.
            ValueError: If 'target_tone' is missing from the context, or if the
                        specified 'target_tone' is not supported by this node.
            RuntimeError: If an unexpected error occurs during the tone
                          conversion process.
        """
        logger.info(f"[{self.node_name}] Starting tone conversion process for input data.")

        if not isinstance(data, str):
            logger.error(f"[{self.node_name}] Invalid input data type. Expected 'str', but received '{type(data).__name__}'.")
            raise TypeError(
                f"Input 'data' for {self.node_name} must be a string. Received type: {type(data).__name__}."
            )

        target_tone = context.get('target_tone')
        if target_tone is None:
            logger.error(f"[{self.node_name}] 'target_tone' parameter is missing from the context.")
            raise ValueError(
                f"'target_tone' must be provided in the context for {self.node_name}."
            )
        if not isinstance(target_tone, str):
            logger.error(f"[{self.node_name}] 'target_tone' in context must be a string. Received type: '{type(target_tone).__name__}'.")
            raise TypeError(
                f"'target_tone' in context for {self.node_name} must be a string. Received type: {type(target_tone).__name__}."
            )

        normalized_tone = target_tone.lower().strip()
        converted_text: str = ""

        try:
            logger.debug(f"[{self.node_name}] Attempting to convert text to '{normalized_tone}' tone.")

            # Simulate tone conversion by prepending a descriptive phrase
            if normalized_tone == 'formal':
                converted_text = f"It is formally conveyed that: \"{data}\""
            elif normalized_tone == 'casual':
                converted_text = f"Yo, check out this casual take: \"{data}\""
            elif normalized_tone == 'professional':
                converted_text = f"From a professional perspective, the content is presented as: \"{data}\""
            elif normalized_tone == 'friendly':
                converted_text = f"Hey there! Here's a friendly version: \"{data}\""
            elif normalized_tone == 'sarcastic':
                converted_text = f"Oh, how absolutely *thrilling*: \"{data}\" (said with utmost sincerity, obviously)."
            else:
                logger.warning(
                    f"[{self.node_name}] Unrecognized target tone: '{target_tone}'. "
                    "Please provide a supported tone."
                )
                raise ValueError(
                    f"Unsupported 'target_tone': '{target_tone}'. "
                    "Supported tones are: 'formal', 'casual', 'professional', 'friendly', 'sarcastic'."
                )

            logger.info(f"[{self.node_name}] Successfully converted text to '{normalized_tone}' tone.")
            return converted_text
        except ValueError:
            # Re-raise ValueErrors related to unsupported tones
            raise
        except Exception as e:
            logger.critical(
                f"[{self.node_name}] An unexpected error occurred during tone conversion: {e}",
                exc_info=True
            )
            raise RuntimeError(
                f"Failed to process tone conversion in {self.node_name} due to an unexpected internal error."
            ) from e