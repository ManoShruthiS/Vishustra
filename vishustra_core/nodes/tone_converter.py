import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverter(BaseNode):
    """
    A processing node designed to simulate converting the tone of input text.

    This node takes a string as input data and a 'target_tone' from the context,
    then applies simple transformations to produce text in the desired tone.
    Supported tones for simulation include 'formal', 'casual', 'sarcastic', and 'empathetic'.
    """

    @property
    def node_name(self) -> str:
        """Returns the programmatic name of this node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data (expected to be a string) and attempts to convert its tone
        based on the 'target_tone' parameter provided in the context dictionary.

        Args:
            data: The input text string that requires tone conversion.
            context: A dictionary containing processing parameters.
                     It MUST include a 'target_tone' key with a string value
                     (e.g., 'formal', 'casual', 'sarcastic', 'empathetic').

        Returns:
            str: The transformed text with the simulated converted tone.

        Raises:
            TypeError: If `context` is not a dictionary.
            ValueError: If `data` is not a string, or if `target_tone` is missing,
                        not a string, or an unsupported value in the context.
            RuntimeError: For unexpected internal errors during the conversion process.
        """
        if not isinstance(context, dict):
            logger.error("Context provided to ToneConverter is not a dictionary. Type: %s", type(context))
            raise TypeError("Context must be a dictionary.")

        if not isinstance(data, str):
            logger.error(
                "ToneConverter received non-string data. Expected 'str', got '%s'. Data: %s",
                type(data).__name__, data
            )
            raise ValueError(f"Input data for ToneConverter must be a string, but received {type(data).__name__}.")

        target_tone = context.get("target_tone")
        if not target_tone or not isinstance(target_tone, str):
            logger.error(
                "Missing or invalid 'target_tone' in context for ToneConverter. Context: %s", context
            )
            raise ValueError("Context must contain a 'target_tone' string (e.g., 'formal', 'casual').")

        logger.info(
            "Starting tone conversion for data (first 75 chars): '%s...' to tone: '%s'",
            data[:75], target_tone
        )

        converted_text = data
        try:
            if target_tone == "formal":
                converted_text = data.capitalize()
                converted_text = converted_text.replace("hello", "Greetings").replace("hi", "Greetings")
                converted_text = converted_text.replace("it's", "it is").replace("i'm", "I am")
                if not converted_text.strip().lower().startswith("greetings") and converted_text.strip():
                    converted_text = "Regarding this matter: " + converted_text
            elif target_tone == "casual":
                converted_text = data.lower()
                converted_text = converted_text.replace("it is", "it's").replace("i am", "i'm")
                if not converted_text.strip().lower().startswith("hey there") and converted_text.strip():
                    converted_text = "Hey there, " + converted_text
                converted_text += " just sayin'." if converted_text.strip() else "just sayin'."
            elif target_tone == "sarcastic":
                converted_text = f"Oh, how truly fascinating: {data}. Clearly, this is groundbreaking work."
            elif target_tone == "empathetic":
                converted_text = f"I truly understand your perspective. It seems that: {data}. Please know I'm here to listen."
            else:
                logger.error(
                    "Unsupported target_tone '%s' for ToneConverter. Context: %s",
                    target_tone, context
                )
                raise ValueError(
                    f"Unsupported 'target_tone': '{target_tone}'. "
                    "Supported tones are 'formal', 'casual', 'sarcastic', 'empathetic'."
                )

        except ValueError as ve:
            # Re-raise specific ValueErrors for unsupported tones or other known input issues
            raise ve
        except Exception as e:
            logger.error(
                "An unexpected error occurred during tone conversion for tone '%s': %s",
                target_tone, e, exc_info=True
            )
            raise RuntimeError(f"Failed to convert tone due to an internal error: {e}") from e

        logger.info(
            "Tone conversion completed successfully for tone: '%s'. Result (first 75 chars): '%s...'",
            target_tone, converted_text[:75]
        )

        return converted_text