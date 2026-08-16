import logging
import re
from typing import Any, Dict

# Assuming BaseNode is available in the specified path.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverterNode(BaseNode):
    """
    A Vishustra node that converts the tone of a given text.
    It expects the 'data' to be a string and 'context' to contain
    'target_tone' specifying the desired output tone.

    Supported tones include: "formal", "casual", "sarcastic".
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ToneConverter"

    def _convert_to_formal(self, text: str) -> str:
        """
        Converts text to a formal tone.
        Ensures the first letter is capitalized and ends with a period.
        """
        text = text.strip()
        if not text:
            return ""
        
        # Capitalize the first letter
        text = text[0].upper() + text[1:]
        
        # Ensure it ends with a period, unless it already ends with common punctuation
        if not text.endswith(('.', '!', '?')):
            text += '.'
        
        logger.debug("Text converted to formal tone.")
        return text

    def _convert_to_casual(self, text: str) -> str:
        """
        Converts text to a casual tone.
        Converts to lowercase, removes most punctuation, and appends " lol".
        """
        text = text.strip()
        if not text:
            return ""
            
        # Convert to lowercase and remove non-alphanumeric characters (keep spaces)
        text = re.sub(r'[^\w\s]', '', text).lower()
        
        # Add a casual suffix if not already present
        if not text.endswith(" lol") and text: # Check text to avoid " lol" for empty string
            text += " lol"
            
        logger.debug("Text converted to casual tone.")
        return text

    def _convert_to_sarcastic(self, text: str) -> str:
        """
        Converts text to a sarcastic tone by alternating the case of characters.
        """
        text = text.strip()
        if not text:
            return ""
            
        converted_chars = []
        for i, char in enumerate(text):
            if i % 2 == 0:
                converted_chars.append(char.upper())
            else:
                converted_chars.append(char.lower())
                
        logger.debug("Text converted to sarcastic tone.")
        return "".join(converted_chars)


    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data (expected to be a string) and converts its tone
        based on the 'target_tone' specified in the context.

        Args:
            data: The input text data (expected str).
            context: A dictionary containing processing parameters.
                     Must include 'target_tone' (str) specifying the desired
                     tone, e.g., "formal", "casual", "sarcastic".

        Returns:
            The text with the converted tone. If the specified 'target_tone'
            is not supported, the original data is returned after logging a warning.

        Raises:
            TypeError: If 'data' is not a string.
            ValueError: If 'target_tone' is missing from context or not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"Invalid input data type for ToneConverter. Expected string, "
                f"got {type(data).__name__}."
            )
            raise TypeError(
                f"ToneConverter expects string data, but received {type(data).__name__}."
            )

        target_tone = context.get("target_tone")
        if not isinstance(target_tone, str):
            logger.error(
                f"Missing or invalid 'target_tone' in context. Expected string, "
                f"got {type(target_tone).__name__}."
            )
            raise ValueError(
                "Context must contain a string 'target_tone' for ToneConverter."
            )

        logger.info(f"Attempting to convert text to '{target_tone}' tone.")
        original_data = data # Keep original for logging/fallback

        # Simulate tone conversion based on target_tone
        lower_target_tone = target_tone.lower()
        if lower_target_tone == "formal":
            converted_data = self._convert_to_formal(data)
        elif lower_target_tone == "casual":
            converted_data = self._convert_to_casual(data)
        elif lower_target_tone == "sarcastic":
            converted_data = self._convert_to_sarcastic(data)
        else:
            logger.warning(
                f"Unsupported target_tone: '{target_tone}'. Returning original data."
            )
            converted_data = data # Return original if tone is not supported

        logger.debug(
            f"Tone conversion complete. Original: '{original_data[:75]}{'...' if len(original_data) > 75 else ''}' "
            f"-> Converted: '{converted_data[:75]}{'...' if len(converted_data) > 75 else ''}'"
        )
        return converted_data