
import logging
from typing import Any, Dict

# Assuming BaseNode is correctly exposed at this path within the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverterNode(BaseNode):
    """
    A Vishustra node designed to simulate the conversion of input text to a specified tone.
    This node processes a string and, based on the 'target_tone' provided in the context,
    returns a modified string that aims to reflect that tone.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data (expected to be a string) and attempts to convert its tone
        based on the 'target_tone' specified in the context.

        Args:
            data: The input text string to be converted.
            context: A dictionary containing runtime information, including
                     'target_tone' (e.g., 'professional', 'casual', 'sarcastic', 'formal').

        Returns:
            The tone-converted string. If conversion fails or tone is unsupported,
            the original data might be returned.

        Raises:
            TypeError: If the input data is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"ToneConverterNode received invalid input data type. "
                f"Expected string, got {type(data).__name__}."
            )
            raise TypeError(
                f"ToneConverterNode expects string data, but received {type(data).__name__}."
            )

        target_tone = context.get("target_tone")
        if not isinstance(target_tone, str) or not target_tone.strip():
            logger.warning(
                "No valid 'target_tone' specified in context. "
                "Defaulting to 'neutral' and applying minimal transformation."
            )
            target_tone = "neutral"
        
        target_tone = target_tone.strip().lower()

        logger.info(f"ToneConverterNode initiated processing for target tone: '{target_tone}'.")

        converted_text = data

        try:
            if target_tone == "professional":
                converted_text = self._transform_to_professional(data)
            elif target_tone == "casual":
                converted_text = self._transform_to_casual(data)
            elif target_tone == "sarcastic":
                converted_text = self._transform_to_sarcastic(data)
            elif target_tone == "formal":
                converted_text = self._transform_to_formal(data)
            elif target_tone == "neutral":
                # For neutral, we might just clean up whitespace or apply no significant change
                converted_text = data.strip()
                logger.debug("Target tone is 'neutral', no specific stylistic transformation applied.")
            else:
                logger.warning(
                    f"Unsupported target tone '{target_tone}'. "
                    "No specific tone transformation applied. Returning original data."
                )
                converted_text = data # Fallback to original data if tone is unsupported

        except Exception as e:
            logger.error(
                f"An unexpected error occurred during tone conversion for tone '{target_tone}': {e}",
                exc_info=True
            )
            # In case of an internal error during simulation, return original data
            converted_text = data 
            logger.info("Returning original data due to an error during tone conversion attempt.")

        logger.info(
            f"ToneConverterNode completed processing. "
            f"Output text length: {len(converted_text)} characters."
        )
        return converted_text

    def _transform_to_professional(self, text: str) -> str:
        """Simulates transforming text to a professional tone."""
        logger.debug("Applying professional tone transformation.")
        text = text.replace("hey", "Dear Team,")
        text = text.replace("wanna", "would like to")
        text = text.replace("lol", "")
        text = text.replace("dude", "colleague")
        text = text.replace("!", ".") # Convert exclamations to periods
        if not text.strip().endswith('.'):
            text += '.'
        return f"Regarding your inquiry: {text.strip().capitalize()}"

    def _transform_to_casual(self, text: str) -> str:
        """Simulates transforming text to a casual tone."""
        logger.debug("Applying casual tone transformation.")
        text = text.replace("regarding your inquiry", "so, about that")
        text = text.replace("would like to", "wanna")
        text = text.replace(".", "!") # Convert periods to exclamations
        text = text.replace("colleague", "buddy")
        text = text.replace("Dear Team,", "Hey there,")
        return f"Hey there, {text.strip().lower()}"

    def _transform_to_sarcastic(self, text: str) -> str:
        """Simulates transforming text to a sarcastic tone."""
        logger.debug("Applying sarcastic tone transformation.")
        text = text.replace("good", "*super*")
        text = text.replace("great", "*marvelous*")
        text = text.replace("problem", "minor inconvenience")
        text = text.replace("solution", "*brilliant* idea")
        text = text.replace("!", "!") # Maintain exclamations or add one
        if not text.strip().endswith('?'):
            text += '!' # Add a touch of sarcasm with exclamation
        return f"Oh, how absolutely *thrilling*: {text.strip()}."

    def _transform_to_formal(self, text: str) -> str:
        """Simulates transforming text to a formal tone."""
        logger.debug("Applying formal tone transformation.")
        text = text.replace("hey", "Greetings,")
        text = text.replace("wanna", "desire to")
        text = text.replace("lol", "[chuckle politely]")
        text = text.replace("dude", "esteemed individual")
        text = text.replace("!", ".") # Convert exclamations to periods
        if not text.strip().endswith('.'):
            text += '.'
        return f"It is my distinct pleasure to inform you that: {text.strip().capitalize()}"

