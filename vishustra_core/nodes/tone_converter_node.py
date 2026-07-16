import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverter(BaseNode):
    """
    A processing node that converts the tone of a given text based on the
    specified target tone in the context.

    This node simulates tone conversion by applying simple text transformations.
    In a production environment, this would typically interface with an LLM
    or a specialized NLP service.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input text data to convert its tone.

        Expects `data` to be a string.
        Expects `context` to contain:
        - 'target_tone': str (e.g., 'professional', 'casual', 'sarcastic', 'friendly')

        Args:
            data (Any): The input text to be converted, expected as a string.
            context (Dict[str, Any]): A dictionary containing processing parameters,
                                       including 'target_tone'.

        Returns:
            Any: The tone-converted string.

        Raises:
            TypeError: If the input data is not a string.
            ValueError: If 'target_tone' is missing from context, not a string,
                        or an unsupported tone.
        """
        logger.info(f"[{self.node_name}] Starting tone conversion process.")

        if not isinstance(data, str):
            logger.error(f"[{self.node_name}] Invalid input data type: Expected string, got {type(data).__name__}.")
            raise TypeError(f"Input data for ToneConverter must be a string, but got {type(data).__name__}.")

        target_tone = context.get('target_tone')
        if not isinstance(target_tone, str):
            logger.error(f"[{self.node_name}] Invalid or missing 'target_tone' in context: Expected string, got {type(target_tone).__name__}.")
            raise ValueError(f"'target_tone' must be a string in context, but got {type(target_tone).__name__}.")

        converted_text = data
        original_tone = "neutral" # In a real system, this might be detected

        logger.debug(f"[{self.node_name}] Converting text from '{original_tone}' to '{target_tone}'.")

        # Simulate tone conversion based on target_tone
        # In a real scenario, this would involve a sophisticated NLP model call
        if target_tone.lower() == 'professional':
            converted_text = converted_text.replace("hey", "Greetings")
            converted_text = converted_text.replace("gonna", "going to")
            converted_text = converted_text.replace("ASAP", "as soon as possible")
            if not converted_text.strip().endswith('.'):
                converted_text += "."
            converted_text += "\n\nBest regards,"
        elif target_tone.lower() == 'casual':
            converted_text = converted_text.replace("Greetings", "Hey there")
            converted_text = converted_text.replace("as soon as possible", "ASAP")
            converted_text = "Just wanted to say, " + converted_text.strip()
            if not converted_text.strip().endswith(('.', '!', '?')):
                converted_text += "!"
        elif target_tone.lower() == 'sarcastic':
            converted_text = converted_text.replace(".", " (or whatever, I guess).")
            converted_text = converted_text.replace("!", " (how exciting!).")
            converted_text = converted_text.replace("?", " (as if I care?).")
            converted_text += " Anyway."
        elif target_tone.lower() == 'friendly':
            if not converted_text.strip().endswith(('.', '!', '?')):
                converted_text += "."
            converted_text += " :)"
        else:
            logger.warning(f"[{self.node_name}] Unsupported target tone: '{target_tone}'. Returning original text.")
            raise ValueError(f"Unsupported target tone: '{target_tone}'.")

        logger.info(f"[{self.node_name}] Successfully converted text to '{target_tone}' tone.")
        return converted_text