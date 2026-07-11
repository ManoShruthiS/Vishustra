
import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverter(BaseNode):
    """
    A processing node that simulates converting the tone of input text
    based on a specified target tone in the context.

    This node demonstrates the structure for text transformation,
    using simple string manipulations to simulate tone changes.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, converting its tone based on the 'target_tone'
        specified in the context.

        Supported tones for simulation include: 'formal', 'informal', 'friendly',
        'professional', and 'neutral'.

        Args:
            data (Any): The input data, expected to be a string containing the text
                        to be converted.
            context (Dict[str, Any]): A dictionary containing contextual information.
                                      Must include 'target_tone' (str) indicating
                                      the desired output tone.

        Returns:
            Any: The tone-converted string.

        Raises:
            TypeError: If the input data is not a string.
            ValueError: If 'target_tone' is missing from the context or an
                        unsupported 'target_tone' is provided.
        """
        logger.debug(f"[{self.node_name}] Starting tone conversion process.")

        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. "
                f"Expected string, got {type(data).__name__}."
            )
            raise TypeError(f"Input data for {self.node_name} must be a string.")

        target_tone = context.get("target_tone")
        if not target_tone:
            logger.error(f"[{self.node_name}] 'target_tone' not found in context.")
            raise ValueError("Missing 'target_tone' in context for ToneConverter.")

        target_tone = str(target_tone).lower()
        supported_tones = {"formal", "informal", "friendly", "professional", "neutral"}

        if target_tone not in supported_tones:
            logger.error(
                f"[{self.node_name}] Unsupported target tone '{target_tone}'. "
                f"Supported tones are: {', '.join(supported_tones)}."
            )
            raise ValueError(
                f"Unsupported target tone '{target_tone}'. "
                f"Supported tones are: {', '.join(supported_tones)}."
            )

        converted_text = str(data)  # Start with a mutable copy of the input text

        # --- Simple Tone Conversion Simulation ---
        # These are basic string manipulations for demonstration purposes.
        # A real-world implementation would involve advanced NLP techniques.
        if target_tone == "formal":
            converted_text = converted_text.replace("I'm", "I am")
            converted_text = converted_text.replace("you're", "you are")
            converted_text = converted_text.replace("don't", "do not")
            converted_text = converted_text.replace("can't", "cannot")
            converted_text = f"Regarding your inquiry: {converted_text.strip()}. Kindly note."
        elif target_tone == "informal":
            converted_text = converted_text.replace("I am", "I'm")
            converted_text = converted_text.replace("you are", "you're")
            converted_text = converted_text.replace("do not", "don't")
            converted_text = converted_text.replace("cannot", "can't")
            converted_text = f"Hey! {converted_text.strip()} Later!"
        elif target_tone == "friendly":
            converted_text = converted_text.replace("thank you", "thanks")
            converted_text = converted_text.replace(".", "! ")
            converted_text = converted_text.replace(";", ", ")
            converted_text = f"Hello there! {converted_text.strip()} Hope you're doing great!"
        elif target_tone == "professional":
            # Focus on clarity and directness, removing excessive casualness.
            converted_text = converted_text.replace("awesome", "excellent")
            converted_text = converted_text.replace("cool", "effective")
            converted_text = converted_text.replace("stuff", "materials")
            converted_text = f"In a professional capacity, please consider: {converted_text.strip()}. Best regards."
        # 'neutral' tone implies no specific transformations beyond the initial text.

        logger.info(f"[{self.node_name}] Successfully converted tone to '{target_tone}'.")
        return converted_text

