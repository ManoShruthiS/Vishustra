import logging
from typing import Any, Dict

# Assuming BaseNode is available at this path within the Vishustra framework
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverter(BaseNode):
    """
    A Vishustra processing node designed to convert the tone of input text.

    This node dynamically adjusts the linguistic style of a given text based on
    a 'target_tone' specified in the processing context. In a production
    environment, this would typically interface with an advanced LLM or NLP
    service to perform sophisticated stylistic transformations. For this
    implementation, a simulated transformation is performed.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the unique name of this processing node.
        """
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data, converting its tone as specified by the context.

        Args:
            data (Any): The input data, expected to be a string (text) that
                        requires tone conversion.
            context (Dict[str, Any]): A dictionary containing parameters for
                                       processing. Expected to include:
                                       - 'target_tone' (str): The desired tone
                                         (e.g., 'professional', 'friendly',
                                         'sarcastic', 'casual', 'neutral').

        Returns:
            str: The text with its tone adjusted according to the 'target_tone'.

        Raises:
            ValueError: If 'data' is not a string or if 'context' is not a dictionary.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input type. Expected 'str' for data, "
                f"but received '{type(data).__name__}'."
            )
            raise ValueError(f"ToneConverter expects string data, but received {type(data).__name__}.")

        if not isinstance(context, dict):
            logger.error(
                f"[{self.node_name}] Invalid input type. Expected 'dict' for context, "
                f"but received '{type(context).__name__}'."
            )
            raise ValueError(f"ToneConverter expects a dictionary for context, but received {type(context).__name__}.")

        target_tone = context.get('target_tone', 'neutral').lower()
        
        # Log the initiation of the conversion process, truncating long inputs for readability
        data_preview = data[:70] + '...' if len(data) > 70 else data
        logger.debug(
            f"[{self.node_name}] Initiating tone conversion for input text: "
            f"'{data_preview}' to target tone: '{target_tone}'."
        )

        transformed_data: str

        # Simulate tone conversion based on the specified target tone
        if target_tone == 'professional':
            transformed_data = f"Regarding the following matter: {data}. Please consider this information carefully."
            logger.info(f"[{self.node_name}] Text successfully converted to a professional tone.")
        elif target_tone == 'friendly':
            transformed_data = f"Hey there! Just wanted to share this with you: {data}. Hope you find it useful!"
            logger.info(f"[{self.node_name}] Text successfully converted to a friendly tone.")
        elif target_tone == 'sarcastic':
            transformed_data = f"Oh, how utterly *fascinating*. You'll just love this: {data}. Truly groundbreaking."
            logger.info(f"[{self.node_name}] Text successfully converted to a sarcastic tone.")
        elif target_tone == 'casual':
            transformed_data = f"So, check this out: {data}. Pretty cool, right?"
            logger.info(f"[{self.node_name}] Text successfully converted to a casual tone.")
        elif target_tone == 'neutral':
            transformed_data = data  # No specific stylistic changes for neutral
            logger.info(f"[{self.node_name}] Target tone set to 'neutral'. No explicit stylistic changes applied.")
        else:
            logger.warning(
                f"[{self.node_name}] Unrecognized 'target_tone': '{target_tone}'. "
                f"Defaulting to original text without tone modification."
            )
            transformed_data = data  # Fallback to the original data for unsupported tones

        logger.debug(f"[{self.node_name}] Tone conversion process completed.")
        return transformed_data
