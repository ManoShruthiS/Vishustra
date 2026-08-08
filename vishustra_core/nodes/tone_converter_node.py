import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverterNode(BaseNode):
    """
    A processing node designed to simulate the conversion of text tone
    based on a specified target tone provided in the context.

    Expected input:
    - data (str): The text string to be converted.
    - context (Dict[str, Any]): Must contain 'target_tone' (str) specifying
      the desired tone (e.g., "formal", "informal", "playful").
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, simulating a tone conversion based on the
        'target_tone' in the provided context.

        Args:
            data (Any): The input data, expected to be a string.
            context (Dict[str, Any]): A dictionary containing additional
                                      parameters, must include 'target_tone'.

        Returns:
            Any: The processed data (string with simulated tone).

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If 'target_tone' is missing or invalid in the context.
        """
        logger.debug("ToneConverterNode received data for processing.")

        if not isinstance(data, str):
            logger.error(
                f"ToneConverterNode: Invalid data type. Expected 'str', "
                f"but received '{type(data).__name__}'."
            )
            raise TypeError(
                f"ToneConverterNode requires string data. "
                f"Received '{type(data).__name__}'."
            )

        target_tone = context.get("target_tone")

        if not isinstance(target_tone, str) or not target_tone:
            logger.error(
                "ToneConverterNode: 'target_tone' (str) is missing or invalid "
                "in the processing context."
            )
            raise ValueError(
                "Context must contain a valid 'target_tone' string "
                "for ToneConverterNode."
            )

        target_tone_lower = target_tone.lower()
        processed_data = data

        # Simulate tone transformation. In a real scenario, this would
        # involve an LLM call or more sophisticated NLP.
        if target_tone_lower == "formal":
            processed_data = f"Please be advised: {data}. We appreciate your understanding."
            logger.info("Data transformed to a formal tone.")
        elif target_tone_lower == "informal":
            processed_data = f"Hey there! Just wanted to share: {data}. Cheers!"
            logger.info("Data transformed to an informal tone.")
        elif target_tone_lower == "playful":
            processed_data = f"Ooh la la! Guess what? {data} - isn't that just peachy?!"
            logger.info("Data transformed to a playful tone.")
        elif target_tone_lower == "neutral":
            # For neutral, we might just pass through or make minimal adjustments
            logger.info("Data maintained a neutral tone (no specific transformation applied).")
            processed_data = data
        else:
            logger.warning(
                f"ToneConverterNode: Unknown target tone '{target_tone}'. "
                "Returning original data without transformation."
            )
            processed_data = data # Fallback to original if tone is not recognized

        logger.debug(
            f"ToneConverterNode finished processing with target tone "
            f"'{target_tone}'. Result length: {len(processed_data)}."
        )
        return processed_data
