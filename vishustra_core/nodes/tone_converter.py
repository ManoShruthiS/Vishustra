import logging
from typing import Any, Dict

# Assuming BaseNode is available at this path in the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverter(BaseNode):
    """
    A processing node that converts the tone of a given text.
    It simulates tone transformation based on a specified target tone in the context.
    This node serves as an example for text manipulation tasks within Vishustra.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, converting its tone based on the 'target_tone'
        specified in the context.

        Args:
            data: The input text to be converted. Expected to be a string.
            context: A dictionary containing operational context, including
                     'target_tone' (str) which dictates the desired output tone.
                     Supported tones include: 'formal', 'casual', 'empathetic', 'concise'.

        Returns:
            The tone-converted string. If the target tone is unsupported or an error
            occurs during conversion, the original data is returned after logging.

        Raises:
            ValueError: If 'data' is not a string, or if 'target_tone' is missing
                        from context or is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                "ToneConverter received non-string data. Expected 'str', got '%s'.",
                type(data)
            )
            raise ValueError(f"ToneConverter requires string data, but received {type(data)}.")

        if 'target_tone' not in context:
            logger.error("ToneConverter context is missing 'target_tone'.")
            raise ValueError("Context must contain 'target_tone' for ToneConverter.")

        target_tone = context['target_tone']
        if not isinstance(target_tone, str):
            logger.error(
                "ToneConverter 'target_tone' in context is not a string. Expected 'str', got '%s'.",
                type(target_tone)
            )
            raise ValueError(f"ToneConverter 'target_tone' must be a string, got {type(target_tone)}.")

        # Normalize target tone for case-insensitive matching and stripping whitespace
        normalized_tone = target_tone.lower().strip()
        converted_data = data # Initialize with original data

        try:
            if normalized_tone == 'formal':
                # Basic formal conversion: Capitalize first letter, ensure ends with period
                temp_data = data.strip()
                if temp_data:
                    converted_data = temp_data[0].upper() + temp_data[1:]
                    if not converted_data.endswith(('.', '!', '?')):
                        converted_data += '.'
                logger.debug("Converted data to formal tone.")
            elif normalized_tone == 'casual':
                # Basic casual conversion: Lowercase, remove trailing punctuation, add a friendly suffix
                temp_data = data.lower().strip()
                if temp_data and temp_data[-1] in ('.', '!', '?'):
                    temp_data = temp_data[:-1] # Remove trailing punctuation
                converted_data = temp_data + ' :)'
                logger.debug("Converted data to casual tone.")
            elif normalized_tone == 'empathetic':
                # Basic empathetic conversion: Prepend an empathetic phrase
                converted_data = "I understand that " + data.strip()
                logger.debug("Converted data to empathetic tone.")
            elif normalized_tone == 'concise':
                # Basic concise conversion: Remove common filler words and extra spaces
                filler_words = ['just', 'very', 'really', 'actually', 'in order to', 'you know', 'a lot of']
                temp_data = data.lower()
                for word in filler_words:
                    temp_data = temp_data.replace(word, '')
                # Reconstruct by splitting and joining to handle multiple spaces from replacements
                converted_data = ' '.join(temp_data.split()).strip()
                # Capitalize first letter of the concise output for better readability
                if converted_data:
                    converted_data = converted_data[0].upper() + converted_data[1:]
                logger.debug("Converted data to concise tone.")
            else:
                logger.warning(
                    "Unsupported target tone '%s' for ToneConverter. Returning original data.",
                    target_tone
                )
                return data # Return original data if tone is unsupported

        except Exception as e:
            # Catch any unexpected errors during the string manipulation
            logger.exception("An unexpected error occurred during tone conversion for tone '%s': %s", target_tone, e)
            # In case of an error, it's often safer to return the original data to avoid breaking the pipeline
            return data

        logger.info(
            "Successfully processed data (first 50 chars: '%s...') into '%s' tone.",
            data[:50].replace('\n', ' '), normalized_tone
        )
        return converted_data
