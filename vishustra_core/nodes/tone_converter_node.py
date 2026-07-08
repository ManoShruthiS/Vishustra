import logging
from typing import Any, Dict, Literal

# Import BaseNode from the specified Vishustra core path
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverterNode(BaseNode):
    """
    A Vishustra processing node responsible for simulating the conversion
    of input text into a specified tone (e.g., formal, informal, neutral).

    This node expects a string as input data and utilizes the 'target_tone'
    parameter from the context dictionary to determine the desired output style.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, transforming its stylistic tone based on the
        'target_tone' specified in the context.

        Args:
            data: The input text (expected to be a string) that needs tone conversion.
            context: A dictionary containing parameters for processing.
                     Must include 'target_tone' (str) with values like 'formal',
                     'informal', or 'neutral'.

        Returns:
            The modified text (str) reflecting the target tone.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If 'target_tone' is missing from `context` or its value
                        is not recognized.
        """
        # --- Input Data Validation ---
        if not isinstance(data, str):
            error_message = f"ToneConverterNode received invalid data type. Expected 'str', got '{type(data).__name__}'."
            logger.error(error_message)
            raise TypeError(error_message)

        # --- Context Parameter Extraction and Validation ---
        target_tone: Literal["formal", "informal", "neutral"]
        try:
            raw_target_tone = context.get("target_tone")
            if raw_target_tone is None:
                raise ValueError("Missing 'target_tone' in context for ToneConverterNode.")
            if not isinstance(raw_target_tone, str):
                raise ValueError(f"Invalid type for 'target_tone'. Expected 'str', got '{type(raw_target_tone).__name__}'.")

            target_tone = raw_target_tone.lower()
            if target_tone not in ["formal", "informal", "neutral"]:
                raise ValueError(f"Unsupported 'target_tone': '{raw_target_tone}'. Expected 'formal', 'informal', or 'neutral'.")
        except ValueError as ve:
            logger.error(f"Context configuration error in ToneConverterNode: {ve}")
            raise ve
        except Exception as e:
            error_message = f"An unexpected error occurred while processing 'target_tone' in context: {e}"
            logger.error(error_message)
            raise ValueError(error_message) # Re-raise as ValueError for context issues

        logger.debug(f"ToneConverterNode processing text (truncated): '{data[:100]}...'")
        logger.info(f"Converting text to '{target_tone}' tone.")

        converted_text = data.strip() # Start with stripped original text

        # --- Simulated Tone Conversion Logic ---
        if target_tone == "formal":
            # Basic simulation: capitalize first letter, ensure sentence ends with a period,
            # replace common informalities with more formal equivalents.
            if converted_text and not converted_text[0].isupper():
                converted_text = converted_text[0].upper() + converted_text[1:]
            if converted_text and not converted_text.endswith(('.', '!', '?')):
                converted_text += '.'
            converted_text = converted_text.replace("hi", "Greetings").replace("hey", "Hello")
            converted_text = converted_text.replace("lol", "").replace("ikr", "Indeed.").replace("gonna", "going to")
            converted_text = converted_text.replace("wanna", "want to").replace("stuff", "items")
            logger.debug("Applied formal tone transformations.")

        elif target_tone == "informal":
            # Basic simulation: lowercase, remove some punctuation, introduce contractions/slang.
            converted_text = converted_text.lower()
            converted_text = converted_text.replace(".", "").replace(",", "").replace("!", "").replace("?", "")
            converted_text = converted_text.replace("hello", "hey").replace("greetings", "hi")
            converted_text = converted_text.replace("i am", "i'm").replace("you are", "you're")
            converted_text = converted_text.replace("going to", "gonna").replace("want to", "wanna")
            # A very simple addition to indicate informality
            if len(converted_text) > 10 and not converted_text.endswith("!"):
                converted_text += " btw"
            logger.debug("Applied informal tone transformations.")

        elif target_tone == "neutral":
            # For neutral, perform minimal changes, mainly stripping whitespace.
            # No significant stylistic changes are applied beyond basic text hygiene.
            logger.debug("Tone set to neutral. No specific tone-modifying transformations applied.")

        logger.info(f"Tone conversion to '{target_tone}' completed successfully.")
        return converted_text