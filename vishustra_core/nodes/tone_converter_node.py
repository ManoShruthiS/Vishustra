import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverterNode(BaseNode):
    """
    A processing node designed to convert the tone of a given text.
    It expects a string as input data and requires a 'target_tone' parameter
    in the context dictionary to determine the desired output tone.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ToneConverterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data (text) to convert its tone based on the
        'target_tone' specified in the context.

        Args:
            data: The input text, expected to be a string.
            context: A dictionary containing processing parameters.
                     Must include 'target_tone' (e.g., 'formal', 'casual', 'neutral').

        Returns:
            The text with its tone converted to the target tone.

        Raises:
            ValueError: If `data` is not a string, or if 'target_tone' is
                        missing from the context.
            Exception: For any unexpected errors during tone conversion.
        """
        logger.debug(
            "ToneConverterNode received data of type %s and context: %s",
            type(data),
            context
        )

        if not isinstance(data, str):
            logger.error(
                "ToneConverterNode expects string data for processing. Received type: %s",
                type(data)
            )
            raise ValueError("Input data for ToneConverterNode must be a string.")

        target_tone_raw = context.get("target_tone")
        if not target_tone_raw:
            logger.error(
                "Required parameter 'target_tone' missing in context for ToneConverterNode."
            )
            raise ValueError(
                "The 'target_tone' parameter is mandatory in the context "
                "for ToneConverterNode processing."
            )

        target_tone = str(target_tone_raw).lower()
        original_text = data
        converted_text = original_text

        try:
            if target_tone == "formal":
                converted_text = self._convert_to_formal(original_text)
                logger.info("Text converted to 'formal' tone.")
            elif target_tone == "casual":
                converted_text = self._convert_to_casual(original_text)
                logger.info("Text converted to 'casual' tone.")
            elif target_tone == "neutral":
                # No change needed for neutral tone.
                logger.info("Target tone 'neutral' requested. No tone conversion applied.")
                converted_text = original_text
            else:
                logger.warning(
                    "Unsupported target_tone '%s' provided. Returning original text without modification.",
                    target_tone
                )
                converted_text = original_text

        except Exception as e:
            logger.error(
                "An unexpected error occurred during tone conversion to '%s': %s",
                target_tone, e, exc_info=True
            )
            # Re-raise the exception to indicate a critical processing failure
            raise

        logger.debug(
            "ToneConverterNode successfully processed data. Original length: %d, Converted length: %d",
            len(original_text), len(converted_text)
        )
        return converted_text

    def _convert_to_formal(self, text: str) -> str:
        """
        Simulates converting text to a more formal tone.
        This is a placeholder for a more sophisticated conversion mechanism.
        """
        if not text:
            return ""
        
        # A very basic simulation: capitalize sentences, add formal greetings/closings
        sentences = text.split('.')
        formal_parts = []
        for sentence in sentences:
            stripped = sentence.strip()
            if stripped:
                formal_parts.append(stripped[0].upper() + stripped[1:] if stripped[0].islower() else stripped)
        
        formatted_text = ". ".join(formal_parts).replace("i am", "I am").replace("i'm", "I am")
        
        if not formatted_text.strip().startswith("Dear"):
            formatted_text = "Dear Sir/Madam, " + formatted_text
        if not formatted_text.strip().endswith("Regards."):
            formatted_text += " Regards."
            
        return formatted_text

    def _convert_to_casual(self, text: str) -> str:
        """
        Simulates converting text to a more casual tone.
        This is a placeholder for a more sophisticated conversion mechanism.
        """
        if not text:
            return ""
            
        # A very basic simulation: lowercasing, adding casual phrases, removing formal elements
        casual_text = text.lower()
        casual_text = casual_text.replace("dear sir/madam", "hey there")
        casual_text = casual_text.replace("regards", "cheers")
        casual_text = casual_text.replace("sincerely", "later")
        casual_text = casual_text.replace(".", "...").replace("!", "!!!").replace("?", "???")
        
        if not casual_text.strip().startswith("hey"):
            casual_text = "hey! " + casual_text
        
        return casual_text