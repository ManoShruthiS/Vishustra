import logging
from typing import Any, Dict

# Assuming BaseNode is correctly located in the project structure as specified
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverterNode(BaseNode):
    """
    A Vishustra processing node that simulates converting the tone of input text.

    This node expects a string 'data' and a 'target_tone' in the 'context' dictionary.
    Supported tones are 'formal', 'informal', and 'sarcastic'. The conversion
    is simulated through basic string manipulations.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of this node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to convert its tone based on the provided context.

        Args:
            data: The input text to be converted. Expected to be a string.
            context: A dictionary containing processing parameters.
                     Must include 'target_tone' (str) with values like
                     'formal', 'informal', 'sarcastic'.

        Returns:
            The tone-converted string. Returns the original data if the
            conversion fails due to an unexpected error or if an unsupported
            tone is specified.

        Raises:
            TypeError: If 'data' is not a string.
            ValueError: If 'target_tone' is missing from context.
        """
        if not isinstance(data, str):
            logger.error(f"{self.node_name}: Received non-string data of type {type(data).__name__}. Expected str.")
            raise TypeError(f"{self.node_name} expects 'data' to be a string, but received {type(data).__name__}.")

        target_tone_raw = context.get("target_tone")
        if not target_tone_raw:
            logger.error(f"{self.node_name}: 'context' is missing the mandatory 'target_tone' key.")
            raise ValueError(f"Context must contain 'target_tone' for {self.node_name}.")
        
        # Ensure target_tone is a string and standardize to lowercase for comparison
        target_tone = str(target_tone_raw).lower()

        original_text = data.strip() # Start with stripped text for cleaner processing
        converted_text = original_text

        logger.info(f"{self.node_name}: Attempting to convert text to '{target_tone}' tone.")

        try:
            if target_tone == "formal":
                # Simulate formalization: capitalize first letter, replace informal greetings, add formal closing.
                if converted_text: # Ensure not empty before accessing index
                    converted_text = converted_text[0].upper() + converted_text[1:]
                converted_text = converted_text.replace("hi", "Dear Sir/Madam").replace("hello", "Dear Sir/Madam")
                if not converted_text.endswith(('.', '!', '?')): # Ensure proper punctuation before appending
                    converted_text += '.'
                converted_text += " Kind regards."
            elif target_tone == "informal":
                # Simulate informalization: lowercase first letter, replace formal greetings, add informal closing.
                if converted_text: # Ensure not empty before accessing index
                    converted_text = converted_text[0].lower() + converted_text[1:]
                converted_text = converted_text.replace("Dear Sir/Madam", "Hey").replace("Kind regards.", "Cheers!")
                if converted_text.endswith('.'): # Remove trailing period if it was added formally
                    converted_text = converted_text.rstrip('.')
                converted_text += " Cheers!"
            elif target_tone == "sarcastic":
                # Simulate sarcasm: A real LLM would rephrase intelligently, but for this simulation,
                # we append a common sarcastic indicator.
                converted_text = original_text + " (obviously...)"
            else:
                logger.warning(
                    f"{self.node_name}: Unsupported target tone '{target_tone}' specified. "
                    "Returning original data without conversion."
                )
                return original_text
            
            logger.info(f"{self.node_name}: Successfully converted text tone to '{target_tone}'.")
            return converted_text

        except Exception as e:
            logger.error(
                f"{self.node_name}: An unexpected error occurred during tone conversion: {e}", 
                exc_info=True
            )
            # In case of an unforeseen error during string manipulation, return original data
            return original_text