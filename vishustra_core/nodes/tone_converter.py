import logging
from typing import Any, Dict
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverter(BaseNode):
    """
    A processing node designed to convert the tone of input text.

    This node simulates tone conversion based on a specified 'target_tone'
    provided in the context dictionary. It supports various common tone
    transformations for text data.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, applying a simulated tone conversion
        based on the 'target_tone' specified in the context.

        Args:
            data: The input data, expected to be a string containing the text
                  whose tone needs to be converted.
            context: A dictionary containing operational context. It must include
                     'target_tone' (str) to specify the desired output tone
                     (e.g., "formal", "informal", "shouting", "whispering",
                     "sarcastic", "neutral").

        Returns:
            The processed data (string) with the simulated tone applied.
            If the 'target_tone' is unrecognized, the original data is returned
            after logging a warning.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If 'target_tone' is missing from the context or is not a string.
        """
        if not isinstance(data, str):
            logger.error("ToneConverter received non-string data. Type: %s", type(data))
            raise TypeError(f"ToneConverter expects string data, but received {type(data)}")

        target_tone = context.get('target_tone')
        if not isinstance(target_tone, str):
            logger.error(
                "Context for ToneConverter is missing 'target_tone' or it's not a string. "
                "Context: %s", context
            )
            raise ValueError(f"Context must contain a string 'target_tone'. Received: {target_tone}")

        original_text = data
        processed_text = original_text
        
        # Simulate tone conversion based on common patterns and context
        normalized_tone = target_tone.lower().strip()

        if normalized_tone == "formal":
            processed_text = f"Kindly note: {original_text.capitalize()}."
            logger.debug("Converted tone to 'formal'.")
        elif normalized_tone == "informal":
            processed_text = f"Hey, just so you know: {original_text.lower()}."
            logger.debug("Converted tone to 'informal'.")
        elif normalized_tone == "shouting":
            processed_text = original_text.upper() + "!!!"
            logger.debug("Converted tone to 'shouting'.")
        elif normalized_tone == "whispering":
            processed_text = f"...{original_text.lower()}..."
            logger.debug("Converted tone to 'whispering'.")
        elif normalized_tone == "sarcastic":
            # A simple, illustrative sarcastic conversion (alternating case)
            processed_text = "".join([c.lower() if i % 2 == 0 else c.upper() for i, c in enumerate(original_text)])
            logger.debug("Converted tone to 'sarcastic'.")
        elif normalized_tone == "neutral":
            # For neutral, we might just strip leading/trailing whitespace
            processed_text = original_text.strip()
            logger.debug("Converted tone to 'neutral'.")
        else:
            logger.warning(
                "Unrecognized 'target_tone' '%s'. No specific tone conversion applied. "
                "Returning original text. Context: %s", target_tone, context
            )
            # In a real scenario, an LLM call would handle arbitrary tones,
            # but for this simulation, we return original if unhandled.
            processed_text = original_text 

        logger.info("ToneConverter processed data. Target Tone: '%s'. Result snippet: '%s'",
                     target_tone, processed_text[:50] + "..." if len(processed_text) > 50 else processed_text)
        return processed_text